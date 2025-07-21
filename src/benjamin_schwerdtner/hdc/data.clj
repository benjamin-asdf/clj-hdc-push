(ns benjamin-schwerdtner.hdc.data
  (:require
   [benjamin-schwerdtner.hdc.hd :as hd])
  (:refer-clojure :exclude [sequence nth assoc contains?]))

(defn sequence
  "
  The resuling represenation is similar to the first item.
  "
  [& elements]
  (hd/superposition (map-indexed
                     (fn [idx e] (hd/permute e idx))
                     elements)))

(defn nth [xs idx] (hd/permute-inverse xs idx))

(defn bound-seq
  "
  Returns a seq where each element is permuted by it's index
  and the resulting represenations are bound.

  The resulting representation is not similar to any inputs.
  (Except in the degenerative single item case.)

  This can be used to represent for instance the paths in a tree with depth
  information.

  "
  [& elements]
  (hd/bind (map-indexed (fn [idx e] (hd/permute e idx))
                        elements)))

;; associative
;; lut
;; mapping ?
(defn record
  [& kvps]
  (hd/superposition (map (fn [[k v]] (hd/bind k v)) kvps)))

(defn assoc [m k v] (hd/superposition m (hd/bind k v)))

(defn lookup [m k] (hd/unbind m k))


;; could retrieve the highest similarity value or something
;; like an alist in lisp
#_(defn assim [m k]
    ;;
    ;; argmax
    (hd/similarity k m))


;; non directed edges, is the same as a record
(def graph record)

(defn directed-graph [& edges]
  (hd/superposition
   (map (fn [[a b]] (hd/non-commutative-bind a b)) edges)))

(defn edge-destination [graph source]
  (hd/non-commutative-unbind graph source))

(defn edge-source [graph destination]
  (hd/non-commutative-unbind-reverse graph destination))

;; ------------

(defn finite-state-machine
  "Returns a representation of a FSM,

  tuples: of the form

  [symbol action outcome]


  Given an FSM symbol and action, [[fsm-next]] outputs 'outcome'.

  [[fsm-source-symbol]] and [[fsm-source-action]] the other way around.

  "
  [& tuples]
  (hd/superposition
   (for [[sym action outcome] tuples]
     (hd/non-commutative-bind (hd/bind sym action) outcome))))

(defn fsm-next
  "Given an FSM, symbol and action,

  returns the next state (destination).
    "
  [fsm sym action]
  (hd/non-commutative-unbind fsm (hd/bind sym action)))

(defn fsm-source-symbol [fsm action destination]
  (hd/unbind
   (hd/non-commutative-unbind-reverse fsm destination)
   action))

(defn fsm-source-action [fsm sym destination]
  ;; it is the same
  (fsm-source-symbol fsm sym destination))

;; ------------------------------

(def binary-tree-left-marker (memoize (fn [] (hd/seed 1))))
(def binary-tree-right-marker (memoize (fn [] (hd/seed 1))))

(def tree-path bound-seq)

(defn tree
  "

  Each
  path-value-pairs:

  [ path value ]

  path:

  [ sym1, sym2, sym3 ]

  For example for a binary tree, chose a left and right marker.

  "
  [& path-value-pairs]
  (hd/superposition
   (for [[path value] path-value-pairs]
     (hd/bind (apply tree-path path) value))))

(defn binary-tree-paths
  "Returns the combination of all binary tree paths up to `depth`
  as a superpostion. "
  [depth]
  ;; When you first think you want some cartesian product but
  ;; _computing in superposition_ is like 'I got that swag'.
  (apply tree-path
         ;; [ marker-superposition, marker-superposition, ... ]
         (repeat (inc depth)
                 (hd/superposition (binary-tree-left-marker)
                                   (binary-tree-right-marker)))))

(defn tree-values [tree paths]
  (hd/unbind tree paths))

(defn binary-tree-values
  "Returns the superposition of the binary tree `tree` values up to `depth`."
  [tree depth]
  (tree-values tree (binary-tree-paths depth)))

;; ---------------

;;
;; In programing in superposition, elements are sets.
;; _Everything is a set._
;;

(defn multiset
  "
  Returns a hd that represents a multiset of the inputs.
  This is `superposition`.

  Note: after normalizing the set, the multi-count information is lost. "
  [& elements]
  (hd/superposition elements))

(defn contains?
  "Returns a rough meassure of how often `elm` is in `x`, a multiset.
  "
  [x elm]
  (hd/similarity x elm))

(defn union
  "Returns a hd that represents the union of the inputs.
  This is `superposition`.
  "
  [& sets]
  (hd/superposition sets))

(defn difference
  "Returns an hd where `sets` are removed from `x`."
  [x & sets]
  (hd/superposition x (hd/negative (hd/superposition sets))))

(defn intersection-1
  "Returns more or less the intersection of `sets`.

  Given a `cutoff-n`, which should more or less be the number
  of contributing elementary hypervectors, else this is not correct.
  "
  [cutoff-n sets]
  ;; - could also take the max value and zero out the rest.
  (hd/cutoff (hd/superposition sets) cutoff-n))

(defn intersection
  [& sets]
  (intersection-1 (count sets) (map hd/normalize sets)))

;; ----------------------

(defn approx-eq?
  ([x elm] (approx-eq? x elm 1.0))
  ([x elm threshold]
   (when x (every? #(<= threshold %) (hd/similarity x elm)))))




(comment

  (hd/similarity
   (intersection a b c)
   [a b c (hd/seed)])


  (hd/similarity
   (intersection (multiset a b)
                 (multiset a c))
   [a b c d (hd/seed)])
  ;; tensor([ 0.9976,  0.5106,  0.4906, -0.0112,  0.0010], device='cuda:0')

  ;; --------------


  (let [book [a b c d]
        m (record [a b] [c d])
        query (intersection (multiset a b) (multiset a c))
        out (lookup m query)]
    (hd/similarity b (hd/cleanup out book)))
  ;; tensor([1.], device='cuda:0')










  (hd/similarity (hd/seed 1) (hd/seed 1))
  (hd/similarity (hd/seed 1) (hd/seed 2))
  (hd/similarity a [a b])


  (hd/similarity (hd/superposition [a a a]) a)




  ;; tensor([0.7473, 0.5038, 0.4938, 0.3701, 0.3693], device='cuda:0')

  (hd/similarity
   (binary-tree-left-marker)
   (hd/unbind
    (tree [[(binary-tree-left-marker)] a])
    a))

  (hd/similarity a (hd/unbind (tree [[(binary-tree-left-marker)] a]) (binary-tree-left-marker)))

  (hd/similarity a (hd/unbind (tree [[(binary-tree-left-marker)
                                      (binary-tree-left-marker)] a]) (binary-tree-left-marker)))

  (hd/similarity
   a
   (hd/unbind
    (tree [[(binary-tree-left-marker) (binary-tree-left-marker)] a])
    (tree-path (binary-tree-left-marker) (binary-tree-left-marker))))
  ;; tensor([1.], device='cuda:0')

  (hd/similarity
   (hd/unbind (tree [[(binary-tree-left-marker)
                      (binary-tree-left-marker)] a])
              (binary-tree-paths 1))
   [a b c (hd/seed 1)])
  ;; tensor([0.2469, 0.1216, 0.1203, 0.1222], device='cuda:0')

  (hd/similarity
   (binary-tree-values
    (tree
     [[(binary-tree-left-marker) (binary-tree-left-marker)] a]
     [[(binary-tree-left-marker) (binary-tree-right-marker)] b])
    1)
   [a b c d (hd/seed 1) (hd/seed 1)])

  ;; tensor([0.1216, 0.1216, 0.0601, 0.0611, 0.0575, 0.0617], device='cuda:0')
  )
(comment

  (def a (hd/seed))
  (def b (hd/seed))
  (def c (hd/seed))
  (def d (hd/seed))

  (hd/superposition a b)

  (hd/superposition [a b])

  (hd/similarity
   (hd/unbind
    (hd/bind [a b]) b)
   [a b c d])

  (hd/similarity (hd/unbind (hd/bind a b) b) [a b c d])

  (let [a (hd/seed)
        b (hd/seed)]
    [
     #_(hd/bind a b)
     #_(hd/unbind (hd/bind a b) b)
     #_(torch/nonzero (hd/bind a b))
     #_(torch/nonzero (hd/unbind (hd/bind a b) b))
     #_(torch/nonzero a)
     (hd/similarity (hd/unbind (hd/bind a b) b) [a b])])




  (hd/similarity
   (fsm-next (finite-state-machine [a b c]) a b)
   [a b c d (hd/seed)])
  ;; tensor([0.4913, 0.5080, 1.0000, 0.5049, 0.4988], device='cuda:0')

  (hd/similarity
   (fsm-source-symbol (finite-state-machine [a b c]) b c)
   [a b c d (hd/seed)])
  ;; tensor([1.0000, 0.5031, 0.4913, 0.4932, 0.4875], device='cuda:0')

  (hd/similarity
   (fsm-source-action (finite-state-machine [a b c]) a c)
   [a b c d (hd/seed)])
  ;; tensor([0.5031, 1.0000, 0.5080, 0.4901, 0.4966], device='cuda:0')

  ;; ---------------------------------------------------------------------
  ;; ================ Programming in superposition ================

  (let
      [s1 (hd/seed)
       s2 (hd/seed)
       a1 (hd/seed)
       a2 (hd/seed)
       fsm (finite-state-machine
            ;; the same outcome from 2 sources, 2 actions
            [s1 a1 c]
            [s2 a2 c])]
    ;; now you query for the superposition of the source symbols

      (hd/similarity
       (fsm-source-symbol fsm (hd/superposition a1 a2) c)
       [s1 s2 a1 a2 a b c (hd/seed)]))

  ;; tensor([0.2534, 0.2534, 0.1284, 0.1284, 0.1255, 0.1296, 0.1303, 0.1206],
  ;;      device='cuda:0')

  (let
      [s1 (hd/seed)
       s2 (hd/seed)
       a1 (hd/seed)
       a2 (hd/seed)
       fsm (finite-state-machine
            ;; the same outcome from 2 sources, 2 actions
            [s1 a1 c]
            [s2 a2 c])]
    ;; and in this version, I query only for 1 action
      (hd/similarity
       (fsm-source-symbol fsm a2 c)
       [s1 s2 a1 a2 a b c (hd/seed)]))

  ;; tensor([0.2549, 0.5036, 0.2519, 0.2500, 0.2529, 0.2484, 0.2512, 0.2510],
  ;;        device='cuda:0')

  ;; here is another variation, but many things are possible, once it clicks with you:

  (let
      [s1 (hd/seed)
       s2 (hd/seed)
       a1 (hd/seed)
       a2 (hd/seed)
       fsm (finite-state-machine
            ;; this time the input symbol is the superposition of s1 and s2
            [(hd/superposition s1 s2) a1 c]
            [a b c])]
      (hd/similarity
       (fsm-source-symbol fsm a1 c)
       [s1 s2 a1 a2 a b c (hd/seed)]))

  ;; .. and the outcome of querying with a1 is the superposition of s1 and s2:

  ;; tensor([0.7517, 0.7468, 0.5037, 0.5100, 0.4997, 0.4914, 0.4976, 0.5001],
  ;;      device='cuda:0')

  ;; ---------------------------------------------------------------------

  (let [m (record [a b] [c d])]
    (hd/similarity m [a b c d (hd/seed)]))
  ;; tensor([0.2369, 0.2413, 0.2481, 0.2451, 0.2489], device='cuda:0')

  (let [m (record [a b] [c d])]
    (hd/similarity
     (lookup m b)
     [a b c d (hd/seed)]))
  ;; tensor([0.4846, 0.2458, 0.2396, 0.2390, 0.2389], device='cuda:0')

  (hd/similarity (sequence a b c) [a b c])
  (hd/similarity (nth (sequence a b c) 0) [a b c])
  (hd/similarity (nth (sequence a b c) 1) [a b c])
  (hd/similarity (nth (sequence a b c) 2) [a b c])
  ;; doesn't care, just is not similar to anything
  (hd/similarity (nth (sequence a b c) 3) [a b c])
  ;;
  ;; in programming in superposition, there are no errors. But 'nonsense' output
  ;;
  ;;


  (hd/similarity a a)
  (hd/similarity a (hd/permute (hd/permute-inverse a)))
  (hd/similarity a (sequence a b))
  (hd/similarity a (sequence a b c))
  (hd/similarity (hd/seed 1) (sequence a b c))

  (hd/similarity (sequence a b c) [a b c (hd/seed)])
  ;; tensor([0.7521, 0.5023, 0.5019, 0.4962], device='cuda:0')
  (hd/similarity (nth (sequence a b c) 1) [a b c (hd/seed)])
  ;; tensor([0.5039, 0.7543, 0.5011, 0.4983], device='cuda:0')

  a
  ;; tensor([[-1., -1.,  1., -1.,  1., -1., -1.,  1.,  1.]], device='cuda:0')
  b
  ;; tensor([[-1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.]], device='cuda:0')

  (hd/superposition [a b])
  ;; tensor([-2.,  0.,  0.,  0.,  0.,  0., -2.,  2.,  0.], device='cuda:0')

  (let [x (hd/superposition [a b c (hd/seed 1)])]
    (hd/similarity
     x
     [a b c (hd/seed 1) (hd/seed 1)]))
  ;; tensor([0.5064, 0.5072, 0.5023, 0.3159, 0.3140], device='cuda:0')

  )

;; ------------------------------------------------------------------
;;
;; Ambiguity
;;
;; These are just ideas.
;;
;; =================================
;;

;;


;; similar to 'dropout'
;; 'vanishingly?' - didn't pick it because word is used in ML.

(defn barely
  "Returns an hd that represents `x` to a tiny amount.

  `p`: The amount of x left over after the operation, default 10%.
  "
  ([x] (barely x 0.1))
  ([x p] (hd/drop-rand x (- 1 p))))

(defn mostly
  "Returns an hd that represents the superposition of mostly `a`
  and a little bit of `b`.

  `p`: The amount of `b` leftover.
  "
  ([a b] (mostly a b 0.1))
  ([a b p]
   (hd/superposition a (barely b p))))


;; 'certainly', 'emphatically', strongly, unquestionably..?

(defn definitely
  "Returns an hd where the similarity (dot similarity) to itself is higher by the factor `alpha`.

  Only works as long as you don't normalize your the interm. representation."
  [x alpha]
  (hd/multiply x alpha))

(defn never
  "Returns an hd that represents the absence of `x`."
  [x]
  (hd/negative x))

;; doesn't work for MAP!
(defn everything
  "Returns an hd that is similar to all other vectors.

  NOTE: this works for binary sparse block codes."
  []
  (hd/ones))

(defn nothing
  "Returns an hd that is dissimilar to all other hds."
  [] (hd/zeroes))

(defn nonsense
  "Returns an hd that means nothing."
  [] (hd/seed))

;; not sure

;; ∀
;; (defn always [])

;; ∃
;; (defn sometimes [])
