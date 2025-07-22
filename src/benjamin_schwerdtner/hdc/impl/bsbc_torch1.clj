(ns  benjamin-schwerdtner.hdc.impl.bsbc-torch1
  (:require
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :refer [py. py..] :as py]

   [benjamin-schwerdtner.hdc.prot :as prot]
   [benjamin-schwerdtner.hdc.opts :refer [*default-opts* *torch-device*]]))

(require-python '[torch :as torch])

(alter-var-root #'*torch-device*
                (constantly
                 (if (py.. torch/cuda is_available) :cuda :cpu)))


;; High-Dimensional Computing with Sparse Vectors
;; M. Laiho et.al. 2015

(defn normalize-inputs
  "Convert either a seq of tensors or a single multi-dim tensor to a consistent format"
  [inputs]
  (if (sequential? inputs)
    ;; (torch/stack (vec inputs))
    (torch/cat (vec inputs))
    inputs))


;; simple binary sparse block code implementation
;;

;; A hd is represented by a (dense) vector of binary sparse values.

;;
;; [ 0, 0, 1, 0, ...  ]
;;
;; For a maximally sparse hd, there is 1 bit 'on' for each `block`.

;;
;;
;;   1 bit per block
;; +-----------+-----------------------------------+
;; |           |                                   |
;; |  1, 0, 0  |                                   |
;; |           |                                   |
;; +-----------+-----------------------------------+
;;                                                  N
;; |-----------|
;;  block-size
;;
;;
;;      1      ,  2,            .. block-count
;;
;;

;; the number of dimensions N is given by
;;
;; N = block-size * block-count.


;; typically
;; block-size  = 500
;; block-count = 20
;; N           = 10000

(defn from-indices
  [indices]
  (let [{:bsbc/keys [block-size block-count N]} *default-opts*
        indices (torch/reshape indices [-1 block-count 1])]
    (-> (torch/zeros [(py.. indices (size 0)) block-count block-size]
                     :device
                     *torch-device*)
        (py.. (scatter_ -1 indices 1))
        (torch/reshape [-1 N]))))


(defn rand-indices
  [batch-dim]
  (let [{:bsbc/keys [N block-size block-count]} *default-opts*]
    (torch/randint 0 block-size [batch-dim block-count] :device *torch-device*)))

(defn indices [x]
  (let [{:bsbc/keys [block-size block-count]} *default-opts*]
    (->
     x
     (torch/reshape [-1 block-count block-size])
     (torch/argmax :dim -1))))

(defn seed
  ([] (seed 1))
  ([batch-dim]
   (from-indices (rand-indices batch-dim))))

(defn superposition
  ([a b] (torch/add a b))
  ([inputs]
   (let [{:bsbc/keys [N]} *default-opts*]
     (torch/einsum
      "ij->j"
      (normalize-inputs inputs)))))

(comment
  (torch/einsum "ij->j"
                (torch/tensor [[1 2 3]
                               [1 2 3]])))

(defn bind
  "Returns an hd that is the binding if the inputs.

  The binding can be used to represne role-filler (key-value) pairs.

  [[unbind]].
  "
  ([inputs]
   ;; for alpha = 1, bind is associative and commutative
   (let [{:bsbc/keys [block-size]} *default-opts*]
     (-> (torch/einsum "ij->j"
                       (indices (normalize-inputs inputs)))
         (torch/remainder block-size)
         from-indices)))
  ;; ----------------
  ([a b] (bind a b 1))
  ;; ----------------
  ([a b alpha]
   (let [{:bsbc/keys [block-size]} *default-opts*]
     (-> (torch/add (indices a) (torch/mul (indices b) alpha))
         (torch/remainder block-size)
         from-indices))))

(defn unbind
  "
  Returns an hd where `b` is unbound from `a`.

  The unbind reverses the binding operation,
  it is used to retrieve the unkown value of a role-filler pair.

  Example:

  (let [a (seed) b (seed)]
    [(torch/allclose a (unbind (bind a b) b))
     (similarity
      (unbind (bind a b) b)
      [a b])])
  ;; [true tensor([1., 0.], device='cuda:0')]

  "
  ([inputs]
   (let [[a & rst] (vec (normalize-inputs inputs))]
     (unbind a (bind rst))))
  ([a b] (bind a b -1)))


(defn permute
  ""
  ([inputs] (permute inputs 1))
  ([inputs n]
   (let [{:bsbc/keys [block-size]} *default-opts*]
     (torch/roll inputs (* n block-size)))))

(defn permute-inverse
  ([inputs] (permute-inverse inputs 1))
  ([inputs n] (permute inputs (- n))))

(defn normalize [inputs]
  (from-indices (indices inputs)))

(defn zeroes
  []
  (let [{:bsbc/keys [N]} *default-opts*]
    (-> (torch/zeros [N] :device *torch-device*))))

(defn ones
  []
  (let [{:bsbc/keys [N]} *default-opts*]
    (-> (torch/ones [N] :device *torch-device*))))


(defn unit-vector
  "This is the vector that bound with any vector x produces x.
  "
  []
  (let [{:bsbc/keys [block-count]} *default-opts*]
    (from-indices
     (torch/zeros [block-count] :device *torch-device* :dtype torch/int64))))


(defn negative
  "Returns the vector for which

  holds:

  hd - x = hd + negative(x)

  where + is superposition and - is not implemented substraction.
"
  [x]
  (torch/negative x))

(defn inverse
  "Returns the inverse of the input.

  This is the vector for which holds:

  (1) unit-vector = bind (x (inverse x))


  (let [a (seed)
        b (seed)]
    (torch/allclose
     (unit-vector)
     (bind a (inverse a))))

  => true

  This is very useful, because the following also holds:


  (2) unbind (a b) =  bind (a (inverse b) )



   (let [a (seed)
        b (seed)
        v (bind a b)]
    [(torch/allclose a (unbind v b))
     (torch/allclose a (bind v (inverse b)))])

  => [true true]


  "
  [x]
  (let [{:bsbc/keys [block-size]} *default-opts*]
    (from-indices (torch/remainder (torch/negative (indices x))
                                   block-size))))


;; dot similarity

(defn dot-similarity
  [x book]
  (let [{:bsbc/keys [N]} *default-opts*]
    (torch/einsum
     "ij,ij->i"
     (torch/reshape x [-1 N])
     (torch/reshape (normalize-inputs book) [-1 N]))))

(defn similarity*
  "Returns a similiarity meassure scaled for the block count.

  This typically is between 0 and 1.

  Can be more than 1, if a vector is more than maximally sparse.
  "
  [x book]
  (let [{:bsbc/keys [block-count]} *default-opts*]
    (torch/div (dot-similarity x book) block-count)))

(defn similarity
  "Returns a similiarity meassure scaled for the block count.

  This typically is between 0 and 1.

  Can be more than 1, if a vector is more than maximally sparse.
  "
  [x book]
  ;; to clj list
  (into [] (py.. (similarity* x book) tolist)))


;; ----------------------
;; below are not bsbc specific...


(defn cleanup
  "Given an hd and a book of hds,
  returns the hd with the highest similarity.
  "
  ([hd book] (cleanup hd book 0.2))
  ([hd book threshold]
   (let [book (normalize-inputs book)
         sim (similarity hd book)
         index (torch/argmax sim)]
     (when (<= threshold (py.. (py/get-item sim index) item))
       (py/get-item book index)))))

;; -----------------------------

(defn drop-rand
  [x probability]
  (let [mask (torch/ge (torch/rand_like x) probability)]
    (torch/where mask x 0)))



;; hm, low is not needed
(defn cutoff
  "Return an hd where the values below `low` and above `high`
  are zoroed out."
  [x low high]
  (let [mask-low (torch/ge x low)
        mask-high (torch/le x high)
        x (torch/where mask-low x 0)
        x (torch/where mask-high x 0)]
    x))

;; --------------------------

(defn multiply [x alpha]
  (torch/mul x alpha))

(comment
  (torch/roll (torch/tensor [0 0 1 1 2 2]) 2)
  (similarity (seed) (seed 10))
  (from-indices (torch/randint 0 3 [2 3] :device *torch-device*))


  (py.. (torch/tensor [64]) (repeat (py/->py-tuple [1 3])))
  (py.. (torch/tensor [2]) (repeat (py/->py-tuple [1 3])))

  (binding [*default-opts*
            {:bsbc/N 9 :bsbc/block-count 3 :bsbc/block-size 3}]
    (let [inds (torch/randint 0 3 [2 3] :device *torch-device*)
          a (py/get-item (from-indices inds) 0)
          b (py/get-item (from-indices inds) 1)
          x (bind a b)]
      [ ;; inds
       inds (from-indices inds) #_(indices (from-indices inds)) x
       (indices x) (torch/allclose a (unbind x b))]))

  (binding [*default-opts*
            {:bsbc/N 9 :bsbc/block-count 3 :bsbc/block-size 3}]
    (unit-vector)
    (inverse
     (from-indices
      (py.. (torch/tensor [2] :device *torch-device* :dtype torch/int64)
            (repeat (py/->py-tuple [1 3]))))))


  (torch/negative (torch/tensor [0 1 0]))

  (torch/subtract
   (torch/tensor [2])
   (torch/tensor [0]))

  (torch/subtract
   (torch/tensor [2])
   (torch/tensor [2]))


  (torch/remainder (torch/tensor [-1]) 3)

  (binding
      [*default-opts*
       {:bsbc/N 9 :bsbc/block-count 3 :bsbc/block-size 3}]
      (let [a (seed) b (seed)]
        [b
         (inverse b)
         (indices b)
         (indices (inverse b))
         (torch/allclose a (unbind (bind a b) b))
         (torch/allclose a (bind (bind a b) (inverse b)))]))


  (binding [*default-opts*
            {:bsbc/N 9 :bsbc/block-count 3 :bsbc/block-size 3}]
    (let [a (seed) b (seed 10)] [a b (similarity a b)]))

  (let [a (seed) b (seed)]
    (similarity (normalize (superposition a b)) [a b]))
  (let [a (seed) b (seed)]
    (similarity (superposition a b) [a b]))

  (let [a (seed)
        b (seed 2)]
    (superposition a b)
    ;; not supported
    (similarity
     (superposition a b)
     [a b]))

  (let [a (seed) b (seed)]
    [(indices b)
     (indices a)
     (indices (bind a b))
     (indices (unbind (bind a b) b))
     (torch/nonzero a)
     (torch/nonzero (unbind (bind a b) b))
     (torch/allclose a (unbind (bind a b) b))])

  (let [a (seed)
        b (seed)
        v (bind a b)]
    [(torch/allclose a (unbind v b))
     (torch/allclose a (bind v (inverse b)))])
  [true true])
