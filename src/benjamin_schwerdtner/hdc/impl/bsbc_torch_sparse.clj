(ns benjamin-schwerdtner.hdc.impl.bsbc-torch-sparse
  (:require

   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :refer [py. py..] :as py]

   [benjamin-schwerdtner.hdc.prot :as prot]
   [benjamin-schwerdtner.hdc.opts :refer [*default-opts* *torch-device*]]))

(do
  ;; (require-python '[numpy :as np])
  (require-python '[torch :as torch])
  (require-python '[torch.sparse :as torch.sparse])
  ;; (require '[libpython-clj2.python.np-array])
  )

;;
;; binary sparse block codes, using torch sparse vectors
;;
(defn superposition [inputs]
  (let [{:bsbc/keys [N block-count block-length]} *default-opts*]))

(defn seed [batch-dim]
  (let [{:bsbc/keys [N block-count block-length]} *default-opts*]
    (torch/sparse_coo_tensor
     ;; indices

     (for [bl block-count]
       (rand-int block-length))

     ;; values
     (torch/ones [(* batch-dim block-count)]))))

(defn from-indices [indices]
  (py..

      (torch/sparse_coo_tensor
       (torch/tensor
        [[0 1 0 0]
         [0 0 1 2]])
       (torch/tensor [1 1 1 1]))
      (coalesce)
      (to_dense))


  )

;; (rand-int 3)

(torch/arange 3)
(torch/tensor [[2]])

(defn seed [batch-dim]
  (let [{:bsbc/keys [N block-count block-length]} *default-opts*
        batch-indices (py.. (torch/arange batch-dim)
                        (unsqueeze 1)
                        (repeat 1 block-count)
                        (flatten)
                        )
        block-offsets (py.. (torch/arange block-count)
                        (mul block-length)
                        (unsqueeze 0)
                        (repeat batch-dim 1))
        random-positions (torch/randint 0 block-length [batch-dim block-count])
        col-indices (py.. (torch/add block-offsets random-positions)
                      (flatten))
        indices (torch/stack [batch-indices col-indices])
        values (torch/ones [(* batch-dim block-count)])]
    (torch/sparse_coo_tensor indices values (torch/Size [batch-dim N]))))

(time
 (binding [*default-opts*
           {:bsbc/N 9
            :bsbc/block-count 3
            :bsbc/block-length 3}]
   (seed 2)))
