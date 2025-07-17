(ns benjamin-schwerdtner.hdc.prot)

(defprotocol VSA
  (superposition [this inputs])
  (bind [this inputs])
  (unbind [this inputs])
  (inverse [this inputs])
  (permute
    [this n inputs]
    [this inputs])
  (permute-inverse
    [this n inputs]
    [this inputs])
  (normalize [this inputs])
  (seed [this])
  (empty [this])
  (ones [this])
  (unit-vector [this]))

;; data
(defprotocol HDMap)

;; could implement
;; clojure.lang.Associative etc
;; then the same code could manipulate clj objects and hdc objects



(defprotocol HDSeq)

(defprotocol HDSet)

(defprotocol HDGraph)

;; (defprotocol HDTree)

(defprotocol HDFSM)
