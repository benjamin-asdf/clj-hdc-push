(ns benjamin-schwerdtner.hdc.hd
  (:require
   ;; [benjamin-schwerdtner.hdc.impl.map-torch :as map-impl]


   ;; [benjamin-schwerdtner.hdc.impl.map-torch :as impl]
   [benjamin-schwerdtner.hdc.impl.bsbc-torch1 :as impl]))

(defprotocol VSA
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
  ([] (impl/seed 1))
  ([batch-dim] (impl/seed batch-dim)))

(defn superposition
  ([a b] (impl/superposition a b))
  ([inputs]
   (impl/superposition inputs)))

(defn bind
  ([a b] (impl/bind a b))
  ([inputs] (impl/bind inputs)))

(defn unbind
  ([a b] (impl/unbind a b))
  ([inputs]
   (impl/unbind inputs)))

(defn inverse [inputs]
  (impl/inverse inputs))

(defn negative [inputs]
  (impl/negative inputs))

(defn permute
  ([inputs n] (impl/permute inputs n))
  ([inputs] (impl/permute inputs)))

(defn permute-inverse
  ([inputs n] (impl/permute-inverse inputs n))
  ([inputs] (impl/permute-inverse inputs)))

(defn normalize [inputs]
  (impl/normalize inputs))

(defn zeroes []
  (impl/zeroes))

(defn ones [] (impl/ones))

(defn unit-vector [] (impl/unit-vector))

(defn similarity* [hd book] (impl/similarity* hd book))

(defn similarity [hd book] (impl/similarity hd book))

;; a -> b
(defn non-commutative-bind [a b]
  (bind a (permute b)))

;; given a, get b
(defn non-commutative-unbind [x a]
  (permute-inverse (unbind x a)))

;; given b, get a
(defn non-commutative-unbind-reverse [x b]
  (unbind x (permute b)))

;; ----------------------------

(defn cleanup
  ([hd book] (impl/cleanup hd book))
  ([hd book threshold] (impl/cleanup hd book threshold)))

(defn cutoff
  ;; this - is MAP specific
  ([x v] (impl/cutoff x (- v) v))
  ([x low high] (impl/cutoff x low high)))

(defn drop-rand
  [x probability]
  (impl/drop-rand x probability))

(defn multiply [inputs alpha]
  (impl/multiply inputs alpha))
