import sys
import torch
import torch.nn as nn
import torchaudio


@torch.inference_mode()
def test(model, test_loader, criterion, lm):

    test_loss_sum = 0
    c_ldist_sum, c_ref_len_sum = 0, 0
    w_ldist_sum, w_ref_len_sum = 0, 0

    for batch in progbar(test_loader):
        waves, labels, input_lens, output_lens = batch
        waves, labels = waves.cuda(
            non_blocking=True), labels.cuda(non_blocking=True)

        out = model(waves)  # (batch, n_class, time)

        loss = criterion(out.permute(2, 0, 1), labels, input_lens, output_lens)
        test_loss_sum += loss.item()

        decoded_preds = lm.decode_ctc(out)
        decoded_targets = lm.decode_ids(labels)
        decoded_targets = [t[:len]
                           for t, len in zip(decoded_targets, output_lens)]

        for hypo, ref in zip(decoded_preds, decoded_targets):
            c_ldist_sum += torchaudio.functional.edit_distance(ref, hypo)
            c_ref_len_sum += len(ref)

            hypo_words = ''.join(hypo).split()
            ref_words = ''.join(ref).split()
            w_ldist_sum += torchaudio.functional.edit_distance(ref_words, hypo_words)
            w_ref_len_sum += len(ref_words)

    test_loss = test_loss_sum / len(test_loader)
    cer = c_ldist_sum / c_ref_len_sum
    wer = w_ldist_sum / w_ref_len_sum

    return test_loss, cer, wer


def collate_fun(batch, encode_fn, mode='train'):
    waves = []
    text_ids = []
    input_lengths = []
    output_lengths = []

    if mode == 'train':
        shifts = torch.randn(len(batch)) > 0.

    for i, (wave, _, text, *_) in enumerate(batch):
        if mode == 'train' and shifts[i]:
            wave = wave[:, 160:]
        waves.append(wave[0])
        ids = torch.LongTensor(encode_fn(text))
        text_ids.append(ids)
        input_lengths.append(wave.size(1) // 320)
        output_lengths.append(len(ids))

    waves = nn.utils.rnn.pad_sequence(waves, batch_first=True).unsqueeze(1)
    labels = nn.utils.rnn.pad_sequence(text_ids, batch_first=True)

    return waves, labels, input_lengths, output_lengths


class GreedyLM:
    def __init__(self, vocab, blank_label='*'):        
        self.vocab = vocab
        self.char_to_id = {c: i for i, c in enumerate(vocab)}
        self.blank_label = blank_label     

    def encode(self, text):
        return [self.char_to_id[c] for c in text.lower()]

    def decode_ids(self, ids):
        if ids.ndim == 2: # batch|steps
            return [self.decode_ids(t) for t in ids]
     
        decoded_text = ''.join([self.vocab[id] for id in ids])

        return decoded_text

    def decode_ctc(self, emissions):
        if emissions.ndim == 3: # batch|labels|steps
            return [self.decode_ctc(t) for t in emissions]

        amax_ids = emissions.argmax(0)
        amax_ids_collapsed = torch.unique_consecutive(amax_ids)
        decoded_text = ''.join([self.vocab[id] for id in amax_ids_collapsed])
        decoded_text = decoded_text.replace(self.blank_label, '')

        return decoded_text


def progbar(iterable, length=30, symbol='='):
    """Wrapper generator function for an iterable. 
       Prints a progressbar when yielding an item. \\
       Args:
          iterable: an object supporting iteration
          length: length of the progressbar
    """
    n = len(iterable)
    for i, item in enumerate(iterable):
        steps = length*(i+1) // n
        sys.stdout.write('\r')
        sys.stdout.write(f"[{symbol*steps:{length}}] {(100/n*(i+1)):.1f}%")
        if i == (n-1):
            sys.stdout.write('\n')
        sys.stdout.flush()
        yield item


### TRAINING ###

def save_checkpoint(model, optimizer, scheduler,
                    epoch, n_iter, config, backup=False):
    if backup:
        fname = f'{config.checkpoint_dir}/states_{n_iter}.pth'
    else:
        fname = f'{config.checkpoint_dir}/states.pth'
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'iter': n_iter,
    }, fname)


def test_and_log(model, test_loader, criterion, writer, lm, n_iter):
    test_loss, test_cer, test_wer = test(model, test_loader, criterion, lm)
    writer.add_scalar('test/loss_test', test_loss, n_iter)
    writer.add_scalar('test/cer', test_cer, n_iter)
    writer.add_scalar('test/wer', test_wer, n_iter)

    print(f"loss: {test_loss:.3f} CER: {test_cer:.3f} WER: {test_wer:.3f}")

    return test_loss


def get_norm(parameters, norm_type=2.0):
    total_norm = torch.norm(torch.stack([torch.norm(
        p.grad.detach(), norm_type) for p in parameters if p.grad is not None]), norm_type)
    return total_norm