from jina import DocumentArray
import finetuner
from finetuner.toydata import generate_fashion_match
from fashion import get_model


embed_model = get_model()

data = DocumentArray(list(generate_fashion_match(num_total=5000, num_pos=5, num_neg=5)))

train_data = data[1000:]
eval_data = data[:1000]

embed_model, summary = finetuner.fit(
    embed_model,
    train_data=train_data,
    eval_data=eval_data,
    loss='CosineTripletLoss',
    interactive=False,
    epochs=5,
    batch_size=1024,
)

summary.plot()

finetuner.fit(embed_model, train_data=eval_data, interactive=True)
