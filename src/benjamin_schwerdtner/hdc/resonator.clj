(ns benjamin-schwerdtner.hdc.resonator
  (:require
   [libpython-clj2.require :refer [require-python]]
   [libpython-clj2.python :refer [py. py..] :as py]
   [benjamin-schwerdtner.hdc.data :as hdd]
   [math.combinatorics.combinatorics :refer [cartesian-product]]
   [benjamin-schwerdtner.hdc.hd :as hd]))

(require-python '[torch :as torch])


;; The problem
;;
;; `factorize`, given a hd find the contributing hdvs:

;;
;; 1. Given a hd `x`.
;; 2. Given a list of codebooks, book-0, book-1, book-2, ...
;; 3. Find a hypervector for each book that together produce x, such that
;;
;;
;;           a-0 ∈ book-0, a-1 ∈ book-1, ...
;;
;; and       x = bind(p(a-0,0), p(a-1,1), ... )
;;
;;
;;
;; where `p(x,n)` is the permutation 'exponentiated' to n.
;; i.e. permuted by n times.
;;

;; Note that
;;
;; f = `bound-seq`,
;;
;; f(a,b,c,...) = bind(p(a,0),p(b,1),...)  is given by:
;;

;; [[benjamin-schwerdtner.hdc.data/bound-seq]]



;; The reference factorizer implementation, and the baseline to meassure the following by.
;;

(defn exhaustive-search-factorize
  "Returns a seq of hdvs from `books`,

  such that


            a-0 ∈ book-0, a-1 ∈ book-1, ...

  and       x = bind(p(a-0,0), p(a-1,1), ... )


  where p(x,n) is [[hd/permute]]. I.e. this is the
  [[benjamin-schwerdtner.hdc.data/bound-seq]]

  We say the hdvs are the factors of x and this function factorizes x.

  Returns nil, if no such list of vectors is found.

  This does an exhaustive search over all possible hdv combinations.

  "
  [x books]
  ;; search space is the cartesian product of the hdvs in books
  (->>
   (apply cartesian-product books)
   (some
    (fn [hdvs] (when (hdd/approx-eq? x (apply hdd/bound-seq hdvs)) hdvs)))))
















;; Neural Implications
;; -------------------
;;
;; Biological plausible resonator networks are a potential path
;; to illuminating possible neuronal circuits for factorization.
;; Theorized to play a role in perception, language processing, analogical reasoning.
;; [https://arxiv.org/abs/2106.05268 references]
;;

;;
;; /Binding/ is the theorized process of associating features of the same (physical) object on the fly.
;; Such has movement, size, color, etc.
;; Von der Malsburg 1999 and others proposed synchrony based mechanism between neuronal submodules.
;;
;; My favorite approach is The 'Reader Centric Cell Assembly' (Buzsáki 2011).
;; Neural syntax: cell assemblies, synapsembles and readers.
;; https://pmc.ncbi.nlm.nih.gov/articles/PMC3005627/
;;
;; I have heard 'it is not clear how synchrony contributes to unified experience / attention / mentality'.
;; This is a misconception cleaned up by Buzsáki. Synchrony only has an effect because there are downstream
;; readers (neurons, effectors) for which synchrony does make a difference.
;; Without a reader or downstream effect, synchrony doesn't have an effect.
;;





















(comment


  (let [b1 (hd/seed 2)
        b2 (hd/seed 2)
        ;; calling vec on the codebook is a leaky abstraction from torch here.
        a (nth (vec b1) 0)
        b (nth (vec b2) 0)
        x (hdd/bound-seq a b)
        outcome
        (exhaustive-search-factorize
         x
         [(vec b1) (vec b2)])]
    [(hd/similarity a outcome)
     (hd/similarity b outcome)])
  ;; [[1.0 0.0] [0.0 1.0]]


  )
