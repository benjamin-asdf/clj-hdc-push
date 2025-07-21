(ns benjamin-schwerdtner.hdc.impl.bsbc-torch-sparse
  (:require
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :refer [py. py..] :as py]
   [benjamin-schwerdtner.hdc.prot :as prot]
   [benjamin-schwerdtner.hdc.opts :refer [*default-opts* *torch-device*]]))

;; simple binary sparse block code implementation
;;

;; A hd is represented as a tensor of N indices


;; [ n1, n2, n3, ... ]

;; where each `n` is a natural number in the rage [ 0, block-size ]
;;
;; This represents a sparse hypervector with exactly 1 active bit per block,
;; There are
;;
;; `block-count = N / block-size`   (1)
;;
;; blocks
;;

;; typically
;; block-size  = 500
;; block-count = 20
;; N           = 10000


(defn seed [batch-dim]
  (let [{:bsbc/keys [N block-size block-count]} *default-opts*]
    (torch/randint 0 block-size [batch-dim block-count])))

(defn superposition [a b])


(defn bind
  [a b]
  (let [{:bsbc/keys [block-size]} *default-opts*]
    (-> (torch/add a b)
        (torch/remainder block-size))))


(comment

  (torch/randint 0 3 [3 3])

  (binding [*default-opts*
            {:bsbc/N 9
             :bsbc/block-size 3
             :bsbc/block-count 3}]
    (seed 3))

  (binding [*default-opts*
            {:bsbc/N 9
             :bsbc/block-size 3
             :bsbc/block-count 3}]
    (let [a (seed 2)
          b (seed 2)]
      [a b
       (superposition a b)])))
