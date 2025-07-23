(ns benjamin-schwerdtner.hdc.analogy
  (:require [benjamin-schwerdtner.hdc.hd :as hd])
  (:refer-clojure :exclude [find]))

;;
;; Analogy
;;

;; ============================================

;;
;; Hofstadter:
;; - Analogies are the essences of situations.
;; - A concept is a city in concept space, it's outskirts are the /concept halo/.
;; - During development, the concept cities of a child grow.
;; - Copernicus (1543): The moon concept becomes the concept of 'moons'.
;;   Moon is a general class of a planet-satilite relationships, of which earth-luna was the prototype.
;; - Such processes are 'concept maturation'.
;;
;;
;; Observations:
;; - planet-satilite relationships: The essence of the situation, the structure of the analogy.
;; - Minksy frame:
;;   skeletton:    { :planet , :satilite }
;;   filled:       { :planet earth, :satilite luna }
;;
;; - Analogies tolerate a certain amount of 'conceptual slippage' (Hofstadter)
;;
;;
;; Analogies on analogies:
;;
;;    Mother    : child      :: Planet : satalite
;;    Container : containand :: Planet : satalite
;;
;; - sort of fits, given a certain amount of slippage.
;;
;;
;;
;;
;; Modeling Analogies
;; -------------------------------
;;
;; - An Analogy model should model:
;; - a concept space,
;; - a prototype operator (HDC: 'seed')
;; - concept maturation operators... ?
;;   (the holy grail)
;; - A strong concept structure model, 'what matters' in a situation (the holy grail... ),
;;   Minsky: Frame
;; - given a new situation: concept retrieval, mapping and inference (see below).
;; - Implement concept halo
;; - Implement slippage (but philosophical question, what is this slippage?)
;;   Naive idea: slippage of an analogy is higher, the larger the concept halos are that are mapped in the analogy.
;;   -> too naive, because slippage can also allow to completely flip a part of an analogy structure.
;;      Only 1 flip out of an otherwise 'good fit' might constitute small slippage.
;;
;;
;;
;;
;;
;; Concepts should be a primitive datastructure
;; --------------------------------
;;
;; - I would expect a concept to be a primitive datastructure in a cognitive system, into which
;;   submodules hook into.
;; - For instance a language module should be able to map concepts to words, etc.
;; - Neocortex is a structure that seems to support such a 'everything coming together' notion.
;; - Neuronal Ensembles (DeNo, Hebb, Abeles, Braitenberg, Yuste, Pulvermueller, ...)
;;   are argued to be the 'neuronal letters' (Buzsáki) of the brain.
;; - And are a candidate for a fundamental datatype of cognition.
;;


;;
;; Perception:
;; ---------------------
;; - Translating a sitation to a point/region in concept space. (psychology: 'categorization')
;; - Since I'm a fan of 'The Brain Inside Out' (Buzsáki 2019),
;;   I imagine the concepts themselves to have an active role.
;;   A system that has a default behaviour, and internal dynamics -
;;   As opposed to a system that translates, given inputs.
;; - Sensor data in this view is more of a resource to the already active perceptions.
;;



;; https://arxiv.org/html/2411.08684v1

;; 'Analogies are partial similarities between different situations
;; that support further inferences' [Gentner 1998]

;;

;;
;; Eqn 1)   A : B :: C : X
;;

;; retrieval
;; -----------
;; Given C find A and B


;; mapping
;; -----------
;; determine the structural correspondance (frame) between A and B
;; find X by applying it to C.
;;
;; A : B  is doing a lot of work... 👈
;;



;; inference
;; ---------------
;; Use A to advance the concept C


;;
;; CST: Conceptual Spaces Theory
;;
;; https://en.wikipedia.org/wiki/Conceptual_space
;;
;; A concept space is similar to a word2vec high dimensional space.
;;
;; - points denote objects
;; - regions is a concept
;;
;; HDC is a natural fit to model CST!
;;


;; crucial insight: that the HDC framework could operationalize the
;; cognitive science theory of Conceptual Spaces [Gärdenfors 2000]
;;


;; =================================================================
;;
;; MODEL: Conceptual Hyperpsace (A CST implemented with HDC).
;;
;; =================================================================
;;
;;

;; Mapping

;; A : B :: C : X

;; determine the structural correspondance (frame) between A and B
;; find X by applying it to C.

;; overall:

;; points in the basis domain
;;
;;
;;
;;
;; point-a ∈ A
;; point-B ∈ B
;; ...



(declare encode find decode)

(defn mapping [points books]
  (let [hdvs (encode points)
        x (find points books)]
    ;; point-x
    (decode x)))

;; -------------------------------------
;; Example D = Color domain,
;; basis: hue, saturation, brightness,
;; k = 3
;; -------------------------------------
;;
;;
;; 1. normalize and put into levels for values
;;
;;    [hue,       saturation,   brightness]
;;
;; => hue: val in range [-10,10] with step 0.5
;;    saturation:
;;    brightness:
;;

;; 2. We obtain 3 code books for Hue,Sat,Bright., each with 40 hdvs standing for the levels
;;

(defn encode [[hue saturation brightness]]

  )



;; I'm not sure why they say 41 buckets?
;; I count 40
(count (range -10 10 0.5))






;; For analogies confined to ojbect categories,
;; CST recommends the Parallelogram model
;; [Rumelhart and Abrahamson, 1973]
(defn find [a b c]

  ;; (c ⊙ a^-1 ) ⊙ b

  )


;;
(def codebook (memoize (fn [resolution k])))

(defn codebook [n] (hd/seed n))

(def codebooks
  (memoize
   (fn [resolution k]
     ;; e.g.
     ;; resolution = 40 (40 levels, buckets for the domain values).
     ;; k = 3  —  Hue, Saturation, Brightness
     (into []
           (repeatedly k #(codebook resolution))))))





;;
(defn decode [x resolution]


  )


(defn prototype [x] x)















;;
;; ==============================
;;
;; 'What Is The Dollar In Mexico' as a special case of a CST implementation.
;;
;; ==============================
;;








;; The CST domain `D` is a vector space with `k` bases
;; For examle the color domain has 3 basis: hue, saturation, brightness (k=3)
;;
;; point-red ∈ D, and 'shades of red' <= D, is a subspace of of D.
;; pont-red is a prototype, is a vector in D.
;;


;; Encode:
;; ---------------------

;; 1. choose basis hypervectors for each k

































































(defn k-combinator [])

;; ??
;; the S combinator

(defn s-combinator [x y z]
  (hd/bind
   (hd/bind x z)
   (hd/bind y z)))



;; (defn implication [a p]
;;   (hd/non-commutative-bind a p))

;; (defn conjunction [a b]
;;   (hd/bind a b))

;; (defn disjunction [a b]
;;   (hd/superposition a b))

;; (defn negation [x]
;;   )
