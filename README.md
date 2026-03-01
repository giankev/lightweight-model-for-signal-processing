# lightweight-model-for-signal-processing

Repository per esperimenti di modelli lightweight per signal processing.

## Notebook `dataset_comunication.ipynb`

Il notebook ora include un'estensione DL OFDM per **raffinare gli LLR** in ingresso al decoder LDPC, usando:

- dataset e baseline già presenti nel notebook (senza ricreare una pipeline esterna);
- training supervisionato con loss bit-level + proxy BLER;
- confronto test baseline vs LLR raffinati con metriche BER/BLER e grafici.

Configurazione inclusa: dataset OFDM da **24k sample** e training multi-epoca.
