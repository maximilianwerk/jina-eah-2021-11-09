import finetuner
from finetuner.toydata import generate_fashion_match
import torch


def get_model():
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(
            in_features=28 * 28,
            out_features=128,
        ),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=128, out_features=32),
    )


def main():
    embed_model = get_model()
    data = generate_fashion_match(num_total=1000)

    finetuner.fit(embed_model, train_data=data, interactive=True)


if __name__ == '__main__':
    main()
