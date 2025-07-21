(ns benjamin-schwerdtner.hdc.resonator
  (:require
   [benjamin-schwerdtner.hdc.data :as hdd]
   [math.combinatorics.combinatorics :refer [cartesian-product]]
   [benjamin-schwerdtner.hdc.hd :as hd]))

;; The problem
;;
;; `factorize`, given x find the contributing hdvs (a,b,c).

;; - You bind hdvs x = bind(a, permute(b,1) , permute(c,2),...), you want to
;;
;;
;; 1. Given a list of codebooks, book-0, book-1, book-2, ...
;; 2. Given a hd `x`.
;; 3. Find which hypervectors in each book that produce x, such that
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

(defn exhaustive-search-factorize [x books]
  ;; search space is the cartesian product of the hdvs in books
  (->>
   (apply cartesian-product books)
   (some
    (fn [hdvs] (when (hdd/approx-eq? x (apply hdd/bound-seq hdvs)) hdvs)))))













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
