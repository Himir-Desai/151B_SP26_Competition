# **Tokens vs Speed and Accuracy**

## **Baseline**

| Sr. No | MAX_TOKENS | max_model_len | max_num_seqs | max_num_batched_tokens | thinking_budget | Input size | Time Taken | Accuracy |
| ------ | ---------- | ------------- | ------------ | ---------------------- | --------------- | ---------- | ---------- | -------- |
| 1      | 4096       | 8192          | 4            | 8192                   | NA              | 64         | 60:11      | 39.06    |

## **thinking_Budget**

| Sr. No | MAX_TOKENS | max_model_len | max_num_seqs | max_num_batched_tokens | thinking_budget | Input size | Time Taken | Accuracy  |
| ------ | ---------- | ------------- | ------------ | ---------------------- | --------------- | ---------- | ---------- | --------- |
| 1      | 4096       | 4096          | 4            | 8192                   | 256             | 64         | 28:59      | 45.31     |
| 2      | 4096       | 4096          | 4            | 8192                   | 512             | 64         | 29:32      | 43.75     |
| 3      | 4096       | 4096          | 4            | 8192                   | **1024**        | 64         | **34:26**  | **57.81** |

## **MAX_TOKENS** ( + optimal thinking_budget)

| Sr. No | MAX_TOKENS | max_model_len | max_num_seqs | max_num_batched_tokens | thinking_budget | Input size | Time Taken | Accuracy  |
| ------ | ---------- | ------------- | ------------ | ---------------------- | --------------- | ---------- | ---------- | --------- |
| 1      | 2048       | 4096          | 4            | 8192                   | 1024            | 64         | 35:02      | 39.06     |
| 2      | 3072       | 4096          | 4            | 8192                   | 1024            | 64         | 34:07      | 45.31     |
| 3      | **4096**   | 4096          | 4            | 8192                   | 1024            | 64         | **33:48**  | **54.69** |
| 4      | 8192       | 8192          | 4            | 8192                   | 1024            | 64         | 39:02      | 56.25     |

## **max_model_len** = MAX_TOKENS + 1024

## **max_num_batched_tokens** ( + optimal thinking_budget + optimal MAX_TOKENS)

| Sr. No | MAX_TOKENS | max_model_len | max_num_seqs | max_num_batched_tokens | thinking_budget | Input size | Time Taken | Accuracy  |
| ------ | ---------- | ------------- | ------------ | ---------------------- | --------------- | ---------- | ---------- | --------- |
| 1      | 4096       | 5120          | 4            | 2048                   | 1024            | 64         | 43:11      | 48.44     |
| 2      | 4096       | 5120          | 4            | **4096**               | 1024            | 64         | **34:17**  | **50.00** |
| 3      | 4096       | 5120          | 4            | 8192                   | 1024            | 64         | 35:13      | 51.56     |
| 4      | 4096       | 5120          | 4            | 16334                  | 1024            | 64         | 34:05      | 50.00     |

## **max_num_seqs** ( + optimal thinking_budget + optimal MAX_TOKENS + optimal max_num_batched_tokens)

| Sr. No | MAX_TOKENS | max_model_len | max_num_seqs | max_num_batched_tokens | thinking_budget | Input size | Time Taken | Accuracy  |
| ------ | ---------- | ------------- | ------------ | ---------------------- | --------------- | ---------- | ---------- | --------- |
| 1      | 4096       | 5120          | 2            | 4096                   | 1024            | 64         | 76:21      | 52.12     |
| 2      | 4096       | 5120          | 4            | 4096                   | 1024            | 64         | 34:56      | 48.44     |
| 3      | 4096       | 5120          | 8            | 4096                   | 1024            | 64         | 25:28      | 53.12     |
| 4      | 4096       | 5120          | **16**       | 4096                   | 1024            | 64         | **16:26**  | **48.44** |
| 4      | 4096       | 5120          | 32           | 4096                   | 1024            | 64         | 16:38      | 46.88     |
