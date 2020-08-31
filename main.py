from model import *
from data import *
from torch.autograd import Variable

criterion = torch.nn.CrossEntropyLoss()

# Now we create the instance of our defined network
net = Net()
# Set the network works in GPU
#net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)#, momentum=0.9)

# losses collection, used for monitoring over-fit
train_losses = []
valid_losses = []

max_epochs = 8
itr = 0

for epoch_idx in range(0, max_epochs):
    for train_batch_idx, (train_input, train_label) in enumerate(train_data_loader):

        itr += 1

        # switch to train model
        net.train()

        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        train_input = Variable(train_input)#.cuda())  # use Variable(*) to allow gradient flow
        train_out = net.forward(train_input)  # forward once

        # compute loss
        train_label = Variable(train_label)#.cuda())
        loss = criterion(train_out, train_label)

        # do the backward and compute gradients
        loss.backward()

        # update the parameters with SGD
        optimizer.step()

        # Add the tuple of （iteration, loss) into `train_losses` list
        train_losses.append((itr, loss.item()))

        if train_batch_idx % 200 == 0:
            print('Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, loss.item()))

        # Run the validation every 200 iteration:
        if train_batch_idx % 200 == 0:
            net.eval()  # [Important!] set the network in evaluation model
            valid_loss_set = []  # collect the validation losses
            valid_itr = 0

            # Do validation
            for valid_batch_idx, (valid_input, valid_label) in enumerate(valid_data_loader):
                net.eval()
                valid_input = Variable(valid_input)#.cuda())  # use Variable(*) to allow gradient flow
                valid_out = net.forward(valid_input)  # forward once

                # compute loss
                valid_label = Variable(valid_label)#.cuda())
                valid_loss = criterion(valid_out, valid_label)
                valid_loss_set.append(valid_loss.item())

                # We just need to test 5 validation mini-batchs
                valid_itr += 1
                if valid_itr > 5:
                    break

            # Compute the avg. validation loss
            avg_valid_loss = np.mean(np.asarray(valid_loss_set))
            print('Valid Epoch: %d Itr: %d Loss: %f' % (epoch_idx, itr, avg_valid_loss))
            valid_losses.append((itr, avg_valid_loss))

# train_losses = np.asarray(train_losses)
# valid_losses = np.asarray(valid_losses)
#
# plt.plot(train_losses[:, 0],      # iteration
#          train_losses[:, 1])      # loss value
# plt.plot(valid_losses[:, 0],      # iteration
#          valid_losses[:, 1])      # loss value
# plt.show()

net_state = net.state_dict()                                             # serialize trained model
torch.save(net_state, os.path.join(minst_dataset_dir, 'minst_net.pth'))    # save to disk
