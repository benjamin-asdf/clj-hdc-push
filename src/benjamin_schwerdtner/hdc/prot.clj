(ns benjamin-schwerdtner.hdc.prot)

(defprotocol Hd
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
  (-similarity [this hd book]))



;; (-seed [this batch-dim])
;; (-zeroes [this])
;; (-ones [this])
;; (-unit-vector [this])




;; ---------------------

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
