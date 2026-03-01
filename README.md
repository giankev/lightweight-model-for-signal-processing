# lightweight-model-for-signal-processing

Repository per esperimenti di modelli lightweight per signal processing.

## Estensione notebook OFDM

La parte Deep Learning è stata integrata direttamente in `dataset_comunication.ipynb`:

- usa la stessa pipeline già presente nel notebook per simulazione dataset e baseline;
- non ricrea un dataset alternativo;
- aggiunge definizione modello PyTorch, training supervisionato e valutazione;
- verifica che il modello superi la baseline nel caso OFDM.
