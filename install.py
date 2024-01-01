import torch





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

a = {1 : 'yo', 2 : 'hi'}
print(list(a.keys()))
