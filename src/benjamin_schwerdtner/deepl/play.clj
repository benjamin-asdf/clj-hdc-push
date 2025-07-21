;; Direct port of https://github.com/pytorch/examples/blob/main/mnist/main.py

(ns benjamin-schwerdtner.deepl.play
  (:require
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :refer [py. py..] :as py]))

(require-python '[torch :as torch])
(require-python '[torch.nn :as nn])
(require-python '[torch.nn.functional :as F])
(require-python '[torch.optim :as optim])
(require-python '[torchvision :as torchvision])
(require-python '[torchvision.datasets :refer [MNIST]])
(require-python '[torchvision.transforms :as transforms])
(require-python '[torch.optim.lr_scheduler :refer [StepLR]])


    ;; def __init__(self):
    ;;     super(Net, self).__init__()
    ;;     self.conv1 = nn.Conv2d(1, 32, 3, 1)
    ;;     self.conv2 = nn.Conv2d(32, 64, 3, 1)
    ;;     self.dropout1 = nn.Dropout(0.25)
    ;;     self.dropout2 = nn.Dropout(0.5)
    ;;     self.fc1 = nn.Linear(9216, 128)
    ;;     self.fc2 = nn.Linear(128, 10)

    ;; def forward(self, x):
    ;;     x = self.conv1(x)
    ;;     x = F.relu(x)
    ;;     x = self.conv2(x)
    ;;     x = F.relu(x)
    ;;     x = F.max_pool2d(x, 2)
    ;;     x = self.dropout1(x)
    ;;     x = torch.flatten(x, 1)
    ;;     x = self.fc1(x)
    ;;     x = F.relu(x)
    ;;     x = self.dropout2(x)
    ;;     x = self.fc2(x)
    ;;     output = F.log_softmax(x, dim=1)
    ;;     return output

(defn create-net
  "Create model using Sequential that matches Python forward pass exactly"
  []
  ;; Python: conv1 -> relu -> conv2 -> relu -> maxpool -> dropout -> flatten -> fc1 -> relu -> dropout -> fc2 -> logsoftmax
  (nn/Sequential
   ;; First conv block
   (nn/Conv2d 1 32 3 :stride 1) ; conv1
   (nn/ReLU) ; relu after conv1

   ;; Second conv block
   (nn/Conv2d 32 64 3 :stride 1) ; conv2
   (nn/ReLU) ; relu after conv2

   ;; Pooling and regularization
   (nn/MaxPool2d 2) ; max_pool2d
   (nn/Dropout 0.25) ; dropout1

   ;; Flatten and fully connected
   (nn/Flatten 1) ; flatten
   (nn/Linear 9216 128) ; fc1
   (nn/ReLU) ; relu after fc1
   (nn/Dropout 0.5) ; dropout2
   (nn/Linear 128 10) ; fc2
   (nn/LogSoftmax :dim 1)) ; log_softmax
  )

;; hm, the above also is >99%.. whatever
;; https://medium.com/data-science/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
(defn create-cnn-model-beyond-99
  "Create CNN model for MNIST classification"
  []
  (nn/Sequential
   (nn/Conv2d 1 32 5 :stride 1)
   (nn/ReLU)
   (nn/Conv2d 32 32 5 :stride 1)
   (nn/BatchNorm2d 32)
   (nn/ReLU)
   (nn/MaxPool2d 2 :stride 2)
   (nn/Dropout 0.25)

   (nn/Conv2d 32 64 3 :stride 1)
   (nn/ReLU)
   (nn/Conv2d 64 64 3 :stride 1)
   (nn/BatchNorm2d 64)
   (nn/ReLU)
   (nn/MaxPool2d 2 :stride 2)
   (nn/Dropout 0.25)

   (nn/Flatten)
   (nn/Linear 576 256)
   (nn/BatchNorm1d 256)
   (nn/ReLU)
   (nn/Linear 256 128)
   (nn/BatchNorm1d 128)
   (nn/ReLU)
   (nn/Linear 128 84)
   (nn/BatchNorm1d 84)
   (nn/ReLU)
   (nn/Dropout 0.25)
   (nn/Linear 84 10)))

;; Training function from PyTorch example
(defn train [model device train-loader optimizer epoch]
  (py. model train)
  (doseq [[batch-idx [data target]] (map-indexed vector train-loader)]
    (let [data (py. data to device)
          target (py. target to device)]
      (py. optimizer zero_grad)
      (let [output (model data)
            loss (F/nll_loss output target)]
        (py. loss backward)
        (py. optimizer step)
        (when (= 0 (mod batch-idx 10))
          (println (format "Train Epoch: %d [%d/%d (%.0f%%)]\tLoss: %.6f"
                           epoch
                           (* batch-idx (py. data size 0))
                           (py.. train-loader -dataset __len__)
                           (* 100.0 (/ batch-idx (py. train-loader __len__)))
                           (py. loss item))))))))

;; Test function from PyTorch example
(defn test
  [model device test-loader]
  (py. model eval)
  (let [test-loss (atom 0)
        correct (atom 0)]
    (py/with [_ (torch/no_grad)]
             (doseq [[data target] test-loader]
               (let [data (py.. data (to device))
                     target (py.. target (to device))
                     output (model data)
                     test-loss-val
                     (py..
                      (F/nll_loss output target :reduction "sum")
                      item)]
                 (swap! test-loss + test-loss-val)
                 (let [pred (py. output argmax :dim 1 :keepdim false)]
                   (swap! correct +
                          (py.. pred
                                (eq (py.. target (view_as pred)))
                                sum
                                item))))))
    (let [dataset-len (py.. test-loader -dataset __len__)
          avg-loss (/ @test-loss dataset-len)
          accuracy (/ @correct dataset-len)]
      (println
       (format
        "\nTest set: Average loss: %.4f, Accuracy: %d/%d (%.0f%%)\n"
        avg-loss
        @correct
        dataset-len
        (* 100.0 accuracy))))))

(comment
  ;; Define device first
  (def device :cuda)

  (def themodel (let
                    [batch-size 64
                     transform
                     (transforms/Compose
                      [(transforms/ToTensor)
                       (transforms/Normalize [0.1307] [0.3081])])

                     train-dataset (MNIST "data/mnist" :train true :download true :transform transform)
                     train-dataloader ((py.. torch/utils -data -DataLoader) train-dataset :batch_size batch-size :shuffle true)
                     test-dataset (MNIST "data/mnist" :train false :download true :transform transform)
                     test-dataloader ((py.. torch/utils -data -DataLoader) test-dataset :batch_size batch-size :shuffle false)

                     model (create-net)
                     model (py.. model (to device))
                     optimizer (optim/Adadelta (py.. model parameters) :lr 1.0)
                     scheduler (StepLR optimizer :step_size 1 :gamma 0.7)
                     _ (doseq [epoch (range 1 15)]
                         (train model device train-dataloader optimizer epoch)
                         (test model device test-dataloader)
                         (py.. scheduler step))]

                    model))

  (torch/save (py.. themodel state_dict) "mnist_cnn.pt")

  (let [w (torch/load "mnist_cnn.pt")
        m (create-net)
        ;; set tensors

        ]

    )

  )
