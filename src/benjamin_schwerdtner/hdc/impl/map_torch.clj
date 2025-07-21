(ns benjamin-schwerdtner.hdc.impl.map-torch
  (:require
   [libpython-clj2.require :refer [require-python]]
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
    (torch/cat (vec inputs))
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

;; dot similiarity, normalized to -1 and 1
(defn similarity
  "Returns a similarity meassure between -1 and 1.

  `book` is a stack/book of tensors.
  "
  [hd book]
  (let [book (normalize-inputs book)
        {:map/keys [dimensions]} *default-opts*
        book (torch/reshape book [-1 dimensions])
        hd (torch/reshape hd [-1 dimensions])]
    (-> (torch/einsum "ij,ij->i" hd book)
        (torch/div dimensions))))



;; --------------

;; a -> b
(defn non-commutative-bind [a b]
  (bind a (permute b)))

;; given a, get b
(defn non-commutative-unbind [x a]
  (permute-inverse (unbind x a)))

;; given b, get a
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

;; --------------------

(defn cleanup
  "Given an hd and a book of hds,
  returns the hd with the highest similarity.
  "
  ([hd book] (cleanup hd book 0.2))
  ([hd book threshold]
   (let [book (normalize-inputs book)
         sim (similarity hd book)
         index (torch/argmax sim)]
     (when
         (<= threshold (py.. (py/get-item sim index) item))
       (py/get-item book index)))))


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
   (non-commutative-unbind
    (non-commutative-bind a b)
    b)
   [a b (seed 1)])



  (let [book
        (normalize-inputs [a b (seed 1)])
        sim (similarity
             (non-commutative-unbind (non-commutative-bind a b) a)
             [a b (seed 1)])]
    (similarity
     (py/get-item
      book
      (torch/argmax sim))
     b))


  (cleanup a [a b])

  (cleanup a (seed 10))


  (let [book (seed 10)
        x (py/get-item book 0)]
    [(cleanup x book)
     (similarity x (cleanup (superposition x (seed 1)) book))])



  (for [n (range 10)]
    (doall
     (for [p (range 0 1 0.05)]
       (let [book (seed 50)
             x (py/get-item book 0)
             y (cleanup (drop-rand x p) book 0.2)]
         [:p p
          (if-not y
            :not-found
            (if (= 1.0 (py.. (similarity x y) item))
              :found
              :found-wrong))]))))

  ;; can drop 0.75% of the vector and still get it out




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



  (torch/matmul
   (torch/tensor [[1 -1 -1]])
   (py..

       (torch/tensor [[-1 -1 -1]
                      [1 -1 -1]
                      [0 0 0]])
     (transpose -2 -1)))





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









(comment
  (do (System/gc) (py.. torch/cuda empty_cache)))
