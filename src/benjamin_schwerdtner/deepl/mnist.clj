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

(defn create-net
  []
  (nn/Sequential (nn/Conv2d 1 32 3 :stride 1)
                 (nn/ReLU)
                 (nn/Conv2d 32 64 3 :stride 1)
                 (nn/ReLU)
                 (nn/MaxPool2d 2)
                 (nn/Dropout 0.25)
                 (nn/Flatten 1)
                 (nn/Linear 9216 128)
                 (nn/ReLU)
                 (nn/Dropout 0.5)
                 (nn/Linear 128 10)
                 (nn/LogSoftmax 1)))


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


(defn keras-tutorial-net
  []
  (nn/Sequential
   (nn/Conv2d 1 32 3)
   (nn/ReLU)
   (nn/MaxPool2d 2)

   (nn/Conv2d 32 64 3)
   (nn/ReLU)
   (nn/MaxPool2d 2)
   (nn/Flatten)

   (nn/Linear 1600 10)
   (nn/Softmax)


   ;; (nn/Flatten)
   ;; (nn/Dropout 0.5)

   ;; (nn/Linear 1600 10)
   ;; (nn/Softmax)
   ))


;; Training function from PyTorch example
(defn train
  [model device train-loader optimizer epoch]
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
          (println
            (format "Train Epoch: %d [%d/%d (%.0f%%)]\tLoss: %.6f"
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
               (let [data (py. data to device)
                     target (py. target to device)
                     output (model data)
                     test-loss-val
                       (F/nll_loss output target :reduction "sum")]
                 (swap! test-loss + (py. test-loss-val item))
                 (let [pred (py. output argmax :dim 1 :keepdim false)]
                   (swap! correct +
                     (py.. pred (eq target) sum item))))))
    (let [dataset-len (py.. train-loader -dataset __len__)
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

  (let [[data target] (first test-loader)
        data (py. data to device)
        target (py. target to device)
        output (model data)
        ]
    [target
     (py. output argmax :dim 1 :keepdim false)])


  ;; Exact PyTorch MNIST example setup

  ;; Device selection
  (def device :cuda)

  ;; Data transforms (exact from PyTorch example)
  (def transform
    (transforms/Compose
     [(transforms/ToTensor)
      (transforms/Normalize [0.1307] [0.3081])]))

  ;; Datasets
  (def train-dataset
    (MNIST "../data" :train true :download true :transform transform))
  (def test-dataset
    (MNIST "../data" :train false :transform transform))

  ;; Data loaders
  (def train-loader
    (torch.utils.data/DataLoader train-dataset :batch_size 64 :shuffle true))
  (def test-loader
    (torch.utils.data/DataLoader test-dataset :batch_size 1000 :shuffle false))

  ;; Model setup (exact from PyTorch example)
  (def model (->
              (create-net)
              ;; (create-cnn-model-beyond-99)
              ;; (keras-tutorial-net)
              (py. to device)))

  (def optimizer (optim/Adam (py. model parameters) :lr 1.0))
  (def scheduler (StepLR optimizer :step_size 1 :gamma 0.7))

  ;; Training loop (14 epochs like PyTorch example)
  (doseq [epoch (range 1 2)]
    (train model device train-loader optimizer epoch)
    (test model device test-loader)
    (py. scheduler step))


  ;; Save model (optional)
  (torch/save (py. model state_dict) "mnist_cnn.pt")

  (test model device test-loader)


  ;; recreate the model


  (def loaded-model
    (let [w (torch/load "mnist_cnn.pt")
          m1 (create-net)]
      (py.. m1 (load_state_dict w))
      (py.. m1 (to device))))

  (test loaded-model device test-loader))
