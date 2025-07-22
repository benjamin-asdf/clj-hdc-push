(ns benjamin-schwerdtner.hdc.hd
  (:require
   ;; [benjamin-schwerdtner.hdc.impl.map-torch :as map-impl]
   ;; [benjamin-schwerdtner.hdc.impl.map-torch :as impl]
   [benjamin-schwerdtner.hdc.impl.bsbc-torch1 :as impl]))

#_(defprotocol VSA
  (-superposition [this inputs])
  (-bind [this inputs])
  (-unbind [this inputs])
  (-inverse [this inputs])
  (-negative [this inputs])
  (-permute
    [this inputs n]
    [this inputs])
  (-permute-inverse
    [this inputs n]
    [this inputs])
  (-normalize [this inputs])
  (-seed [this batch-dim])
  (-zeroes [this])
  (-ones [this])
  (-unit-vector [this])
  (-similarity [this book hd]))

;; Create implementations map
#_(def implementations
  {:torch-map
   (reify VSA
     (-superposition [this inputs] (map-impl/superposition inputs))
     (-bind [this inputs] (map-impl/bind inputs))
     (-unbind [this inputs] (map-impl/unbind inputs))
     (-inverse [this inputs] (map-impl/inverse inputs))
     (-negative [this inputs] (map-impl/negative inputs))
     (-permute [this inputs] (map-impl/permute inputs))
     (-permute [this inputs n] (map-impl/permute inputs n))
     (-permute-inverse [this inputs n] (map-impl/permute-inverse inputs n))
     (-permute-inverse [this inputs] (map-impl/permute-inverse inputs))
     (-normalize [this inputs] (map-impl/normalize inputs))
     (-seed [this batch-dim] (map-impl/seed batch-dim))
     (-zeroes [this] (map-impl/zeroes))
     (-ones [this] (map-impl/ones))
     (-unit-vector [this] (map-impl/unit-vector))
     (-similarity [this book hd] (map-impl/similarity book hd)))})

(defn seed
  "Returns a fresh hd.

  batch-dim: default 1.
  "
  ([] (impl/seed 1))
  ([batch-dim] (impl/seed batch-dim)))

(defn superposition
  "Returns an hd that represents the superposition of the inputs.

  The resultant hd is similiar to all all the inputs.

  This is like a logical disjunction.
  "
  ([a b] (impl/superposition a b))
  ([inputs]
   (impl/superposition inputs)))

(defn bind
  "Returns an hd that represents the binding of the inputs.

  The resultant hd is dissimilar to all the inputs.

  The binding is used to represent the key-value pair.
  The input can be recovered by unbinding with the other input.

  "
  ([a b] (impl/bind a b))
  ([inputs] (impl/bind inputs)))

(defn unbind
  "Reverse the binding operation."
  ([a b] (impl/unbind a b))
  ([inputs] (impl/unbind inputs)))

(defn inverse
  "Returns an hd that is the inverse of the input.

  Unbinding with an hd is the same as binding with the inverse.
  Binding an hd with its inverse results in the unit vector.

  "
  [inputs]
  (impl/inverse inputs))

(defn negative
  "Returns an hd that is the negative element of superposition.

  a - b =  a + negative(b)
  "
  [inputs]
  (impl/negative inputs))

(defn permute
  "Returns an hd that is permuted (rolled) `n` times.

  The resultant hd is dissimilar to the input.
  It is said to be 'randomizing'.

  The resultant hd is located in a well known, unrelated domain.

  It can be used as the quotation of a symbol, or position marker in a sequence.

  Also called 'protect', 'quote'.

  "
  ([inputs n] (impl/permute inputs n))
  ([inputs] (impl/permute inputs)))

(defn permute-inverse
  "The inverse of [[permute]].

  This is the same as calling permute with negative arg.

  Also called 'unprotect', 'unquote'."
  ([inputs n] (impl/permute-inverse inputs n))
  ([inputs] (impl/permute-inverse inputs)))

(defn normalize
  "Returns an hd with default properties, clamped element values,
  optionally broken ties, default sparsity etc. depending on impl."
  [inputs]
  (impl/normalize inputs))

(defn zeroes
  "Returns an hd that is only zeroes. "
  []
  (impl/zeroes))

(defn ones
  "Returns an hd that is only ones."
  [] (impl/ones))

(defn unit-vector
  "Returns the unit vector.

  The unit vector has the property that it is the neutral element of binding.

  (1)  b * unit-vector = b

  also the following holds:

  (2) b * inverse(b) = unit-vector.

  Binding and superposition with the neutral elements is an algebraic ring or something.
  "
  [] (impl/unit-vector))

(defn similarity*
  "Returns a tensor of similarity values.
  Between `hd` and the hds in `book`.

  Depending on implementation this is normalized between 0 and 1.
  "
  [hd book] (impl/similarity* hd book))

(defn similarity
  "Like [[similarity*]] but returns a clojure vec."
  [hd book] (impl/similarity hd book))

;; a -> b
(defn non-commutative-bind
  "Returns an hd that represents a non-commutative-bind of the inputs.

  This has all properties of bind, but unbinding doesn't go in both directions.
  This can be used to represent a directed edge in a graph.

  It is implemented as

  (bind a (permute b))

  Terminology:

  a : source
  b : target

  edge : source->target
  "
  [a b]
  (bind a (permute b)))

;; given a, get b
(defn non-commutative-unbind
  "Returns an hd that is the target of the directed-edge `x`, given `a` as source.

  See [[non-commutative-unbind]]."
  [x a]
  (permute-inverse (unbind x a)))

;; given b, get a
(defn non-commutative-unbind-reverse
  "Returns an hd that is the source of the directed-edge `x`, given `b` as target."
  [x b]
  (unbind x (permute b)))

;; ----------------------------

(defn cleanup
  "Returns the hd of `book` with the highest similarity to `hd`.

  With threshold > 0, a similarity value (see [[similiarity]]),
  this can also return nil.


  (The min threshold depends on implemenation).

  Default threshold is 0.2.
  "
  ([hd book] (impl/cleanup hd book))
  ([hd book threshold] (impl/cleanup hd book threshold)))

(defn cutoff
  "Return an hd where the values below and above low and high are
  zoroed out.
  This is a kind of context dependend thinning."
  ;; this - is MAP specific
  ([x v] (impl/cutoff x (- v) v))
  ([x low high] (impl/cutoff x low high)))

(defn drop-rand
  "Return an hd with random bits zeroed out

  `probability`: A drop chance between 0 and 1."
  [x probability]
  (impl/drop-rand x probability))

(defn multiply
  "Returns an hd  with element wise multiplication by factor `alpha`.

  This is the same as `superposition` with itself `alpha` times."
  [inputs alpha]
  (impl/multiply inputs alpha))
