from jina import DocumentArray
import finetuner
from finetuner.toydata import generate_fashion_match
from finetuner.embedding import embed
from fashion import get_model

embed_model = get_model()

data = DocumentArray(list(generate_fashion_match(num_total=5000, num_pos=5, num_neg=5)))

train_data = data[1000:]
eval_data = data[:1000]


for doc in data:
    doc.embedding = doc.blob.flatten()

eval_data.plot_embeddings(colored_attr='tags__class', output='blob.png', title='blobs')

embed(eval_data, embed_model)
eval_data.plot_embeddings(
    colored_attr='tags__class',
    output='before_training.png',
    title='before_training',
)

embed_model, summary = finetuner.fit(
    embed_model,
    train_data=train_data,
    eval_data=eval_data,
    loss='CosineTripletLoss',
    interactive=False,
    epochs=5,
    batch_size=1024,
)

embed(eval_data, embed_model)
eval_data.plot_embeddings(
    colored_attr='tags__class', output='after_training.png', title='after_training'
)
summary.plot()

finetuner.fit(embed_model, train_data=eval_data, interactive=True)
