import os
import pandas as pd
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import answer_similarity
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from ragas.metrics import (
    context_precision, 
    answer_relevancy,
    faithfulness,
    context_recall,
    answer_correctness
)

# Set API key for Azure
os.environ["OPENAI_API_KEY"] = ""

# Azure configurations
azure_configs = {
    "base_url": "https://ai.com",
    "model_deployment": "gpt-4-32k-0613",
    "model_name": "gpt-4",
    "embedding_deployment": "text-embedding-ada-002",
    "embedding_name": "text-embedding-ada-002",
}

# Initialize Azure Chat model
azure_model = AzureChatOpenAI(
    openai_api_version="2023-10-01-preview",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["model_deployment"],
    model=azure_configs["model_name"],
    validate_base_url=False,
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Initialize Azure embeddings
azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["embedding_deployment"],
    model=azure_configs["embedding_name"],
)

# Sample data
data_samples = {
    'question': [
        'When was the first super bowl?', 
        'Who won the most super bowls?'
    ],
    'answer': [
        'The first superbowl was held on Jan 15, 1967', 
        'The most super bowls have been won by The New England Patriots'
    ],
    'contexts': [
        [
            'The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'
        ], 
        [
            'The Green Bay Packers...Green Bay, Wisconsin.',
            'The Packers compete...Football Conference'
        ]
    ],
    'ground_truth': [
        'The first superbowl was held on January 15, 1967', 
        'The New England Patriots have won the Super Bowl a record six times'
    ]
}

# List of metrics
metrics = [
    answer_correctness,
    answer_similarity,
    faithfulness,
]

# Function to evaluate metrics for a single row
def evaluate_row(row):
    row_data = {
        'question': [row['question']],
        'answer': [row['answer']],
        'contexts': [row['contexts']],
        'ground_truth': [row['ground_truth']]
    }
    dataset = Dataset.from_dict(row_data)
    result = evaluate(dataset, metrics=metrics, llm=azure_model, embeddings=azure_embeddings)
    return result

# Create a DataFrame from the original data_samples
final_df = pd.DataFrame(data_samples)

# Initialize new columns for each metric
metric_names = ['answer_correctness', 'answer_similarity', 'faithfulness']
for metric_name in metric_names:
    final_df[metric_name] = 0.0

# Apply evaluate_row function to each row and update the DataFrame
for index, row in final_df.iterrows():
    result = evaluate_row(row)
    for metric_name in metric_names:
        final_df.at[index, metric_name] = result[metric_name]

# Save the DataFrame to a CSV file
final_df.to_csv('evaluated_dataset.csv', index=False)

print(final_df)