(ns benjamin-schwerdtner.hdc.impl.bsbc-torch-sparse
  (:require
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :refer [py. py..] :as py]
   [benjamin-schwerdtner.hdc.prot :as prot]
   [benjamin-schwerdtner.hdc.opts :refer [*default-opts* *torch-device*]]))

;; ----------------------------------
;; attempt.
;; index representation or simply dense torch tensors are way easier to implement.
;;


(require-python '[torch :as torch])
(require-python '[torch.sparse :as torch.sparse])

(defn normalize-inputs
  "Convert either a seq of tensors or a single multi-dim tensor to a consistent format"
  [inputs]
  (if (sequential? inputs)
    ;; (torch/stack (vec inputs))
    (torch/cat (vec inputs))
    inputs))

;;
;; Binary Sparse Block Codes using PyTorch sparse tensors
;;

(defn seed
  "Creates a random BSBC hypervector with one active bit per block.
  Can optionally specify batch dimension for multiple vectors."
  ([] (seed 1))
  ([batch-dim]
   (let [{:bsbc/keys [N block-count block-length]} *default-opts*
         ;; Create batch indices [0,0,...,0, 1,1,...,1, ...]
         batch-indices (-> (torch/arange batch-dim)
                           (torch/unsqueeze 1)
                           (py.. (repeat 1 block-count))
                           (torch/flatten))
         ;; Create block offsets [0, block-length,
         ;; 2*block-length, ...]
         block-offsets (py.. (-> (torch/arange block-count)
                                 (torch/mul block-length)
                                 (torch/unsqueeze 0))
                         (repeat batch-dim 1))
         ;; Random position within each block
         random-positions
         (torch/randint 0 block-length [batch-dim block-count])
         ;; Combine to get column indices
         col-indices (-> (torch/add block-offsets random-positions)
                         (torch/flatten))
         ;; Stack to create indices tensor
         indices (torch/stack [batch-indices col-indices])
         ;; All values are 1 for binary vectors
         values (torch/ones [(* batch-dim block-count)])]
     (torch/sparse_coo_tensor indices
                              values
                              (torch/Size [batch-dim N])
                              :device
                              *torch-device*))))



(defn zeroes
  "Creates an empty BSBC hypervector (all zeros)."
  ([] (zeroes 1))
  ([batch-dim]
   (let [{:bsbc/keys [N]} *default-opts*]
     (torch/sparse_coo_tensor (torch/tensor [[] []])
                              (torch/tensor [])
                              (torch/Size [batch-dim N])
                              :device
                              *torch-device*))))


(defn ones
  "Creates a BSBC hypervector with all bits set to 1."
  ([]
   (ones 1))
  ([batch-dim]
   (let [{:bsbc/keys [N]} *default-opts*]
     (torch/ones [batch-dim N] :device *torch-device*))))


(defn superposition
  ([a b]
   (torch/add a b))
  ([inputs]
   (-> (normalize-inputs inputs)
       (torch/sum :dim -2))))


;; (defn bind [a b])



(comment
  (binding [*default-opts*
            {:bsbc/N 9 :bsbc/block-count 3 :bsbc/block-length 3}]
    (py.. (seed 1) (to_dense))
    (let [a (seed 1)
          b (seed 2)]
      [(py.. a (to_dense)) (py.. b (to_dense))
       (py.. (superposition [a b])
         (to_dense))])
    ;; (py.. (zeroes 2) (to_dense))
    )
  (binding [*default-opts*
            {:bsbc/N 9 :bsbc/block-count 3 :bsbc/block-length 3}]
    (py.. (seed 1) (to_dense))
    (let [a (seed 1)
          b (seed 2)]
      [(py.. a (to_dense)) (py.. b (to_dense))
       (superposition [a b])])
    ;; (py.. (zeroes 2) (to_dense))
    ))
