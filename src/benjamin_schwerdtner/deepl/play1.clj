(ns benjamin-schwerdtner.deepl.play1
  (:require
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :refer [py. py..] :as py]
   [benjamin-schwerdtner.hdc.prot :as prot]
   [benjamin-schwerdtner.hdc.opts :refer [*default-opts* *torch-device*]]))

(require-python '[torch :as torch])

(require-python '[torch.nn :as nn])
(require-python '[torch.nn.functional :as F])
(require-python '[torch.optim :as optim])

;; Simple bit-flipping neural network
;; Goal: Learn to flip bits in 1x3 one-hot encoded vectors
;; Input: [1 0 0] -> Output: [0 1 1]

(defn create-training-data
  "Generate training data for bit flipping"
  []
  (let [inputs [[1.0 0.0 0.0]
                [0.0 1.0 0.0]
                [0.0 0.0 1.0]]
        targets [[0.0 1.0 1.0] [1.0 0.0 1.0] [1.0 1.0 0.0]]]
    {:inputs (torch/tensor inputs)
     :targets (torch/tensor targets)}))

(defn create-network "Simple 3->4->3 feedforward network"
  []
  (nn/Sequential
   (nn/Linear 3 4)
   (nn/ReLU)
   (nn/Linear 4 3)
   (nn/Sigmoid)))

(defn train-network "Train the network to flip bits"
  [model data epochs learning-rate]
  (let [optimizer (optim/Adam (py. model parameters) :lr learning-rate)
        loss-fn (nn/MSELoss)]
    (doseq [epoch (range epochs)]
      (let [outputs (model (:inputs data))
            loss (loss-fn outputs (:targets data))]
        (py. optimizer zero_grad)
        ;; backward computes new gradients
        (py. loss backward)
        (py. optimizer step)
        (when (zero? (mod epoch 100))
          (println (format "Epoch %d, Loss: %.4f" epoch (py. loss item))))))
    model))

;; (py.. (nn/Linear 3 4) -weight)

(defn test-network [model]
  "Test the trained network on single inputs"
  (let [test-inputs (torch/tensor [[1.0 0.0 0.0]])]
    (py. (model test-inputs) tolist)))


(comment
  (def model (train-network
              (create-network)
              (create-training-data) 1000 0.1))

  (model (torch/tensor [[1.0 0.0 0.0]]))
  (user/test-learning-rates)
  (println model))
