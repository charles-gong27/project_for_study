import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import average_precision_score, accuracy_score
np.random.seed(2020)
torch.manual_seed(2020)
#classes = ('jumping', 'phoning', 'playinginstrument', 'reading', 'ridingbike', 'ridinghorse', 'running', 'takingphoto', 'usingcomputer', 'walking','others')
#'''
classes = ('applauding', 'blowing_bubbles', 'brushing_teeth',
            'cleaning_the_floor', 'climbing', 'cooking', 'cutting_trees',
            'cutting_vegetables', 'drinking', 'feeding_a_horse',
            'fishing', 'fixing_a_bike', 'fixing_a_car', 'gardening',
            'holding_an_umbrella', 'jumping', 'looking_through_a_microscope',
            'looking_through_a_telescope', 'playing_guitar', 'playing_violin',
            'pouring_liquid', 'pushing_a_cart', 'reading', 'phoning',
            'riding_a_bike', 'riding_a_horse', 'rowing_a_boat', 'running',
            'shooting_an_arrow', 'smoking', 'taking_photos', 'texting_message',
            'throwing_frisby', 'using_a_computer', 'walking_the_dog',
            'washing_dishes', 'watching_TV', 'waving_hands', 'writing_on_a_board', 'writing_on_a_book')
#'''
# Implementing CBOW model for the exercise given by a tutorial in pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
context_size = 3  # {w_i-2 ... w_i ... w_i+2}
embedding_dim = 60


def get_ap_score(y_true, y_scores):
    scores = 0.0
    scores += average_precision_score(y_true, y_scores)
    return scores

f = open('trainwordst.txt')
train_text = []
for line in f:
    line = line.strip('\n')
    line = line.rstrip('\n')
    words = line.split('#')
    for opo in words[2].split():
        train_text.append(opo)
f.close()
fv=open('testwordst.txt')
val_text = []
for line in fv:
    line = line.strip('\n')
    line = line.rstrip('\n')
    words = line.split('#')
    for opo in words[2].split():
        val_text.append(opo)
fv.close()
def make_context_vector(context, word_to_idx):
    idxs = [word_to_idx[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


vocab = set(train_text)
vocab_size = len(vocab)




word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

def nomprebab(a):
    numoo = []
    probab = []
    pronam = []
    for i in classes:
        numoo.append(word_to_idx[i])
    for i in numoo:
        probab.append(a[0][i])
    '''
    for i in probab:
        pronam.append((i - min(probab)+0.1) / (max(probab) - min(probab)+0.1))
    '''
    return probab


data = []
dataval=[]
for i in range(len(train_text) // 3):
    context = [train_text[i * 3 + 0], train_text[i * 3 + 1], ]
    target = train_text[i * 3 + 2]
    data.append((context, target))
for i in range(len(val_text) // 3):
    context = [val_text[i * 3 + 0], val_text[i * 3 + 1], ]
    target = val_text[i * 3 + 2]
    dataval.append((context, target))

class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, 128)
        self.output = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = F.relu(self.proj(embeds))
        out = self.output(out)
        nll_prob = F.log_softmax(out, dim=-1)
        return nll_prob


modelll = CBOW(vocab_size, embedding_dim)
optimizer = optim.SGD(modelll.parameters(), lr=0.001)

losses = []
loss_function = nn.NLLLoss()
criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

for epoch in range(30):
    print('epoch', epoch, 'train start')
    total_loss = 0

    for context, target in data:
        context_vector = make_context_vector(context, word_to_idx)

        # Remember PyTorch accumulates gradients; zero them out
        modelll.zero_grad()

        nll_prob = modelll(context_vector)
        loss = loss_function(nll_prob, Variable(torch.tensor([word_to_idx[target]])))

        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss)
# Let's see if our CBOW model works or not
print("*************************************************************************")


