(ns benjamin-schwerdtner.hdc.impl.bsbc-torch-sparse
  (:require
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :refer [py. py..] :as py]
   [benjamin-schwerdtner.hdc.prot :as prot]
   [benjamin-schwerdtner.hdc.opts :refer [*default-opts* *torch-device*]]))

(require-python '[torch :as torch])
(require-python '[torch.sparse :as torch.sparse])

;;
;; Binary Sparse Block Codes using PyTorch sparse tensors
;;


(defn seed
  "Creates a random BSBC hypervector with one active bit per block.
  Can optionally specify batch dimension for multiple vectors."
  ([]
   (seed 1))
  ([batch-dim]
   (let [{:bsbc/keys [N block-count block-length]} *default-opts*
         ;; Create batch indices [0,0,...,0, 1,1,...,1, ...]
         batch-indices (py.. (torch/arange batch-dim)
                             (unsqueeze 1)
                             (repeat 1 block-count)
                             (flatten))
         ;; Create block offsets [0, block-length, 2*block-length, ...]
         block-offsets (py.. (torch/arange block-count)
                             (mul block-length)
                             (unsqueeze 0)
                             (repeat batch-dim 1))
         ;; Random position within each block
         random-positions (torch/randint 0 block-length [batch-dim block-count])
         ;; Combine to get column indices
         col-indices (py.. (torch/add block-offsets random-positions)
                           (flatten))
         ;; Stack to create indices tensor
         indices (torch/stack [batch-indices col-indices])
         ;; All values are 1 for binary vectors
         values (torch/ones [(* batch-dim block-count)])]
     (torch/sparse_coo_tensor indices values
                              (torch/Size [batch-dim N])
                              :device *torch-device*))))

(defn zoroes
  "Creates an empty BSBC hypervector (all zeros)."
  ([]
   (empty 1))
  ([batch-dim]
   (let [{:bsbc/keys [N]} *default-opts*]
     (torch/sparse_coo_tensor (torch/tensor [[] []])
                              (torch/tensor [])
                              (torch/Size [batch-dim N])
                              :device *torch-device*))))

(defn ones
  "Creates a BSBC hypervector with all bits set to 1."
  ([]
   (ones 1))
  ([batch-dim]
   (let [{:bsbc/keys [N]} *default-opts*]
     ;; (torch/ones [batch-dim ])
     )))
