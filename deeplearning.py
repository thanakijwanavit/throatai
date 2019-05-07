
import torch

# method for validation
def validation(model, validloader, criterion):
    valid_loss = 0
    accuracy = 0
    for images, labels in validloader:

        images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer,validloader, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    if device == 'gpu':
        model.to('cuda')

        for e in range(epochs):
            running_loss = 0
            model.train()
            for ii, (inputs, labels) in enumerate(trainloader):
                steps += 1

                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    #to test the model
                    model.eval()
                    with torch.no_grad():
                        valid_loss, valid_accuracy = validation(model, validloader, criterion)


                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Loss: {:.4f}".format(running_loss/print_every),
                          "Validation Loss: {:.4f}".format(valid_loss/len(validloader)),
                          "Validation Accuracy: {:.4f}".format(valid_accuracy/len(validloader)))

                    running_loss = 0

    else:
        for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(trainloader):
                steps += 1


                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Loss: {:.4f}".format(running_loss/print_every))

                    running_loss = 0
    return model
