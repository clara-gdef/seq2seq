import math

import torch
from allennlp.modules.elmo import batch_to_ids

from search import BeamSearch, Sampling


# adapted from https://github.com/pytorch/fairseq/blob/master/fairseq/sequence_generator.py
class SequenceGenerator(object):
    def __init__(
        self,
        pad, eos, vocab_size,
        search='beam',
        beam_size=1,
        min_len=1,
        max_len=200,
        stop_early=True,
        sampling_topk=-1,
        unk=1,
        unk_penalty=0,
    ):
        self.pad = pad
        self.eos = eos
        self.unk = unk
        self.unk_penalty = unk_penalty
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len = max_len
        self.min_len = min_len
        self.stop_early = stop_early

        if search == 'random':
            self.search = Sampling(pad, eos, vocab_size, sampling_topk)
        if search == 'beam':
            self.search = BeamSearch(pad, eos, vocab_size)

    def generate(self, model, decoder_hidden, model_type, bos_token, hidden=None):
        bsz = 1
        beam_size = self.beam_size
        max_len = self.max_len

        # model.lin_out is a dirty trick to create a tensor on the same machine as the model (GPU or CPU)
        scores = model.lin_out.weight.new(bsz * beam_size, max_len + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = model.lin_out.weight.data.new(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = bos_token
        nonpad_idxs = None

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        worst_finalized = [{'idx': None, 'score': -math.inf} for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfin_idx, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if self.stop_early or step == max_len or unfinalized_scores is None:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                best_unfinalized_score = unfinalized_scores[unfin_idx].max()
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
                unfinalized_scores: A vector containing scores for all
                    unfinalized hypotheses
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            tokens_clone[:, step] = self.eos

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                def get_hypo():

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                elif not self.stop_early and score > worst_finalized[sent]['score']:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]['idx']
                    if worst_idx is not None:
                        finalized[sent][worst_idx] = get_hypo()

                    # find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                    worst_finalized[sent] = {
                        'score': s['score'],
                        'idx': idx,
                    }

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfin_idx, unfinalized_scores):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None
        if decoder_hidden is not None:
            ht, ct = [], []
            for l in range(len(decoder_hidden)):
                hid = decoder_hidden[l]  # (bs, hidden)
                # (bs, 1, hidden) -> (bs, beam_size, hidden)
                hid = hid.expand(
                    bsz, beam_size, hid.shape[-1]).contiguous()
                ht.append(hid.view(bsz * beam_size, hid.shape[-1]))
                c = decoder_hidden[l]  # (bs, hidden)
                # (bs, 1, hidden) -> (bs, beam_size, hidden)
                c = c.expand(
                    bsz, beam_size, c.shape[-1]).contiguous()
                ct.append(c.view(bsz * beam_size, c.shape[-1]))
            # hidden = torch.stack(hidden).to(self.device)
            # for l in range(len(hidden[0])):
                # hid = hidden[0][l]
                # hid = hid.unsqueeze(2).expand(hid.shape[0], bsz, beam_size, hid.shape[2]).contiguous()
                # ht.append(hid.view(hid.shape[0], bsz * beam_size, hid.shape[3]))
                # c = hidden[1][l]
                # c = c.unsqueeze(2).expand(c.shape[0], bsz, beam_size, c.shape[2]).contiguous()
                # ct.append(c.view(c.shape[0], bsz * beam_size, c.shape[3]))
            hid_dec = (ht, ct)

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                ht = [h[:, reorder_state] for h in hid_dec[0]]
                ct = [c[:, reorder_state] for c in hid_dec[1]]
                hid_dec = (ht, ct)

            if model_type == "elmo":
                # decoder step
                character_ids = model.batch_to_ids(tokens[:, step])
                emb = model.elmo(character_ids.cuda())
                # emb = model.word_embedding(tokens[:, step]).unsqueeze(0)
                emb_tensor = emb["elmo_representations"][-1]
                inputs = torch.cat([hidden, emb_tensor], dim=2)
                # TODO: check if this hidden is indeed the hidden state we want
                output, hidden = model.lstm(inputs, hid_dec)
            elif model_type == "ft":
                inputs = torch.cat([hidden, tokens[:, step].type(dtype=torch.FloatTensor).unsqueeze(-1).unsqueeze(-1).cuda()],
                                   dim=2)
                # TODO: check if this hidden is indeed the hidden state we want
                # tmp_hid_state = hid_dec[0][0].unsqueeze(0) if shape[]
                tmp, hidden_dec = model.lstm(inputs, (hid_dec[0][0].unsqueeze(0), hid_dec[1][0].unsqueeze(0)))
                output = model.lin_out(tmp)

            # else:
            #     output, hidden = model.decoder(conds, emb, hidden)
            lprobs = torch.log_softmax(output.squeeze(0), 1).detach()

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)
            if step < max_len:
                # self.search.set_src_lengths(src_lengths)

                cand_scores, cand_indices, cand_beams = self.search.step(
                    step,
                    lprobs.view(bsz, -1, self.vocab_size),
                    scores.view(bsz, beam_size, -1)[:, :, :step],
                )
            else:
                # make probs contain cumulative scores for each hypothesis
                lprobs.add_(scores[:, step - 1].unsqueeze(-1))

                # finalize all active hypotheses once we hit max_len
                # pick the hypothesis with the highest prob of EOS right now
                torch.sort(
                    lprobs[:, self.eos],
                    descending=True,
                    out=(eos_scores, eos_bbsz_idx),
                )
                num_remaining_sent -= len(finalize_hypos(step, eos_bbsz_idx, eos_scores))
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)

            finalized_sents = set()
            if step >= self.min_len:
                # only consider eos when it's among the top beam_size indices
                torch.masked_select(
                    cand_bbsz_idx[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_bbsz_idx,
                )
                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(
                        cand_scores[:, :beam_size],
                        mask=eos_mask[:, :beam_size],
                        out=eos_scores,
                    )
                    finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores, cand_scores)
                    num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                # src_lengths = src_lengths[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                bsz = new_bsz
            else:
                batch_idxs = None

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = buffer('active_mask')
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(_ignore, active_hypos)
            )

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

        return finalized


def generate_sequence(model, decoder_hidden, model_type, ntoken, start_id, end_id, search='greedy', size=1, sampling_topk=-1, max_len=200,
                      hidden=None, unk=1, unk_penalty=0):
    if search == 'greedy':
        search = 'random'
        sampling_topk = 1
        size = 1
    _default_pad_id = 0
    generator = SequenceGenerator(_default_pad_id, end_id, ntoken, search=search, beam_size=size,
                                  max_len=max_len, sampling_topk=sampling_topk, unk=1, unk_penalty=0)
    output = generator.generate(model, decoder_hidden, model_type, start_id, hidden=hidden)
    tokens = [[hyp['tokens'] for hyp in sent] for sent in output]
    nlls = [[hyp['score'] for hyp in sent] for sent in output]
    return tokens, nlls

#
# if __name__ == "__main__":
#     ntoken = 100
#     pad = 0
#     nwe = 50
#     nz = 10
#     bsz = 3
#     beam_size = 5
#     cond = torch.randn(bsz, nz)
#     generator = SequenceGenerator(pad, eos, ntoken, beam_size=beam_size)
#     with torch.no_grad():
#         output = generator.generate(model, bos, cond)
#     print(output[0][0].keys())
#     print(output[0][0]['score'])
