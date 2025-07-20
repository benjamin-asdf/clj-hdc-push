(ns benjamin-schwerdtner.hdc.impl.map-torch
  (:require [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :refer [py. py..] :as py]
            ;; [benjamin-schwerdtner.hdc.prot :as prot]
            [benjamin-schwerdtner.hdc.opts :refer [*default-opts*
                                                   *torch-device*]]))

(require-python '[torch :as torch])

;; Pentti Kanerva's MAP based on the 2009 book chapter
;; Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors. Cognitive Computation, 1(2), 139-159.
;;


(defn normalize-inputs
  "Convert either a seq of tensors or a single multi-dim tensor to a consistent format"
  [inputs]
  (if (sequential? inputs)
    (torch/stack (vec inputs))
    inputs))


(defn seed [batch-dim]
  (let [{:map/keys [dimensions]} *default-opts*]
    (-> (torch/rand [batch-dim dimensions] :device *torch-device*)
        (torch/sub 0.5)
        (torch/sign))))


;; given by:
;; A + negative(B) = A - B
;;
;; where `+` is superposition
;; `-` would be 'substract', the inverse of superposition
(defn negative [hd]
  (torch/mul -1 hd))

(defn normalize [hd]
  (torch/sign hd))

(defn zeroes
  []
  (let [{:map/keys [dimensions]} *default-opts*]
    (-> (torch/zeros [dimensions] :device *torch-device*))))

(defn ones
  []
  (let [{:map/keys [dimensions]} *default-opts*]
    (-> (torch/ones [dimensions] :device *torch-device*))))


(defn superposition
  ([a b] (torch/add a b))
  ([inputs]
   (let [{:map/keys [dimensions]} *default-opts*]
     (-> (normalize-inputs inputs)
         (torch/reshape [-1 dimensions])
         (torch/sum :dim -2)))))

(defn bind
  ([a b] (torch/mul a b))
  ([inputs]
   (let [{:map/keys [dimensions]} *default-opts*]
     (-> (normalize-inputs inputs)
         (torch/reshape [-1 dimensions])
         ;; is commutative
         (torch/prod :dim 0)))))


;; bind is unbind in MAP
(def unbind bind)

;; ... as a consequence, the inverse is identity
(def inverse identity)


(defn unit-vector
  "Returns the unit vector.

  A vector bound to it's inverse returns the unit vector.

  Is the same as `ones` in MAP.
  "
  []
  (ones))


(defn permute
  "Permutes the inputs by a given n.

   The permutation is done by shifting the elements of the input tensor by n positions."
  ([inputs]
   (permute inputs 1))
  ([inputs n]
   (torch/roll inputs n)))


(defn permute-inverse
  "Inverse of the permutation.

  The inverse is done by shifting the elements of the input tensor by -n positions.
  "
  ([inputs]
   (permute-inverse inputs 1))
  ([inputs n]
   (permute inputs (- n))))


    ;; def dot_similarity(self, others: "MAPTensor", *, dtype=None) -> Tensor:
    ;;     """Inner product with other hypervectors"""
    ;;     if dtype is None:
    ;;         dtype = torch.get_default_dtype()

    ;;     if others.dim() >= 2:
    ;;         others = others.transpose(-2, -1)

    ;;     return torch.matmul(self.to(dtype), others.to(dtype))


(defn similarity
  "Returns a similarity meassure between 0 and 1.

  `book` is a stack/book of tensors.

  This looks at the overlap where the sign of the vectors agree.


  "
  [hd book]

  (let [{:map/keys [dimensions]} *default-opts*
        book (-> (normalize-inputs book)
                 (torch/reshape [-1 dimensions])
                 (normalize))
        hd (normalize (torch/reshape hd [dimensions]))]
    (->
     (torch/eq book hd)
     (torch/sum :dim 1)
     (torch/div dimensions)))



  )




(defn non-commutative-bind [a b]
  (bind a (permute b)))

(defn non-commutative-unbind [x a]
  (permute-inverse (unbind x a)))

(defn non-commutative-unbind-reverse [x b]
  (unbind x (permute b)))

;; ------------------------

(defn cutoff
  "Return an hd where the values below `low` and above `high`
  are zoroed out."
  [x low high]
  (let [mask-low (torch/ge x low)
        mask-high (torch/le x high)
        x (torch/where mask-low x 0)
        x (torch/where mask-high x 0)]
    x))


;; -----------------------

(defn drop-rand
  [x probability]
  (let [mask (torch/ge (torch/rand_like x) probability)]
    (torch/where mask x 0)))




(comment

  (drop-rand (torch/rand [10]) 0.1)
  (drop-rand (torch/rand [10]) 1)
  (drop-rand (torch/rand [10]) 0.5)
  ;; tensor([0.1728, 0.4436, 0.0000, 0.0000, 0.0000, 0.9219, 0.6336, 0.0000, 0.0000,
  ;;       0.0000])


  (torch/bernoulli (torch/tensor [10]))

  (torch/where
   ;; (torch/lt (torch/tensor [1 2 3 4 5]) 2)
   (torch/tensor [false true false])
   (torch/tensor [1 2 3])
   0)

  (cutoff (torch/tensor [1 2 3 4 5]) 2 4)


  (def a (seed 1))
  (def b (seed 1))

  (similarity
   (non-commutative-unbind (non-commutative-bind a b) b)
   [a b (seed 1)])

  (similarity
   (non-commutative-unbind (non-commutative-bind a b) a)
   [a b (seed 1)])


  (similarity
   (non-commutative-unbind (non-commutative-bind a b) a)
   [a b (seed 1)])

  (similarity
   (non-commutative-unbind-reverse (non-commutative-bind a b) a)
   [a b (seed 1)])

  (similarity
   (non-commutative-unbind-reverse (non-commutative-bind a b) b)
   [a b (seed 1)]))
















(comment

  (py.. (superposition (seed 1) (seed 1)) -dtype)


  (torch/mv
   (torch/tensor
    [[-1 -1 -1]
     [1 -1 -1]])
   (torch/tensor [1 -1 -1]))



  (torch/mv
   (torch/tensor [[1 -1 -1]])
   (torch/tensor [1 -1 -1]))


  (binding [*default-opts* {:map/dimensions 3}]
    (similarity
     [(torch/tensor [1 1 1])
      (torch/tensor
       [-1 -1 1])]
     (torch/tensor [-1 -1 -1])))


  (binding [*default-opts* {:map/dimensions 3}]
    (superposition
     [(torch/tensor [1 2 3])
      (torch/tensor [1 2 3])]))



  (binding [*default-opts* {:map/dimensions 3}]
    (permute
     [(torch/tensor [1 2 3])
      (torch/tensor [1 2 3])]))


  (binding [*default-opts* {:map/dimensions 3}]
    (permute-inverse
     (permute
      [(torch/tensor [1 2 3])
       (torch/tensor [1 2 3])])))




  (binding [*default-opts* {:map/dimensions 3}]
    (bind
     (torch/tensor [[1 2 3]
                    [1 2 3]]))))
