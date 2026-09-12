import torch
from generator import Generator
from config import parse_opt
import os
import json

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        idx2word = {i: line.strip() for i, line in enumerate(f)}
    return idx2word

def generate_sentences(model, start_token, eos_token, batch_size, idx2word, device):
    model.eval()
    with torch.no_grad():
        samples = model.sample(batch_size, start_token, device)
        sentences = []
        for seq in samples:
            words = [idx2word[idx.item()] for idx in seq if idx.item() in idx2word and idx.item() != 0 and idx2word[idx.item()] != '<START>']
            if '<EOS>' in words:
                eos_index = words.index('<EOS>')
                words = words[:eos_index]
            sentence = " ".join(words)
            sentences.append(sentence)
        return sentences

def main():
    opt = parse_opt()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab_path = os.path.join(opt.save_path, 'vocab.txt')
    idx2word = load_vocab(vocab_path)
    vocab_size = len(idx2word)

    # get the index of <EOS>
    eos_token = list(idx2word.keys())[list(idx2word.values()).index('<EOS>')]

    tweets = []
    classes = []
    o_label = 0

    training_tags = [k for k in opt.SENTIMENT_CLASSES if k != 'nor']
    for tag in training_tags:
        print(f"Generating samples for class: {tag}")
        model = Generator(vocab_size, opt.EMB_DIM, opt.HIDDEN_DIM, opt.MAX_SEQ_LENGTH).to(device)
        model_path = os.path.join(opt.save_path, f'generator_toxic_{tag}.pt')
        model.load_state_dict(torch.load(model_path))
        model.eos_token = eos_token

        sentences = generate_sentences(model, opt.START_TOKEN, eos_token, opt.gen_num, idx2word, device)
        tweets.extend(sentences)
        classes.extend([str(o_label)] * len(sentences))
        o_label += 1

    output_file = os.path.join(opt.save_path, 'data_gen_toxigan.json')
    with open(output_file, 'w') as f:
        json.dump({"tweet": tweets, "class": classes}, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(tweets)} samples to {output_file}")

if __name__ == '__main__':
    main()
