#from main import *
from model import *
from data import *

# Create a net instance
test_net = Net()

# Load serialize data
test_net_state = torch.load(os.path.join(minst_dataset_dir, 'minst_net.pth'))

# Load weights/parameters from serialized data
test_net.load_state_dict(test_net_state)

test_net.eval()

# random pick a test item in `test_set_list`
test_item = random.choice(test_set_list)

test_img_path = os.path.join(minst_dataset_dir, test_item['file_path'])

# Load image as gray-scale image
img = np.asarray(Image.open(test_img_path).convert('L'), dtype=np.float32) / 255.0
h, w = img.shape[0], img.shape[1]
img_tensor = torch.from_numpy(img)

# Reshape to (1, 1, 28, 28), the first 1 set the mini-batch to 1, the second is the channel size
img_tensor = img_tensor.view((1, 1, h, w))

# Forward for prediction
pred = test_net.forward(img_tensor)#.cuda())

# Find the label with max probability
prob_max = torch.argmax(pred.detach(), dim=1)

# Show the result
plt.imshow(img, cmap='gray')
plt.title("Predicted Label %d" % (prob_max.item()))
plt.show()
