(ns benjamin-schwerdtner.hdc.opts)

(def ^:dynamic *default-opts*
  {:bsbc/N (long 1e4)
   :bsbc/block-count 20
   :bsbc/block-length (/ (long 1e4) 20)})

(def ^:dynamic *torch-device* :cuda)
