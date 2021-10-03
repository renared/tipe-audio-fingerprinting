# Audio fingerprinting
TIPE 2018 : Algorithmes de production et identification d’empreintes de sons musicaux

## 1st program

Inspired by [1].

Identifies peaks in 6 frequency bands of the spectrogram at every time step. Computes the difference between the peaks of a fragment and those of a song and identifies the song that minimizes the difference. Really high computation time (increasing really fast with the number of songs in the database) and too much data stored.

## 2nd program

Based on Shazam [2], identifies local peaks in the spectrogram, creates pairs of peaks stored as hashes in a SQLite database, identification performed by a SQL query.

## 3rd program

Based on [3] which uses wavelet decomposition instead of the spectrogram approach, has not been finished.

## References

[1] Roy van Rijn : Creating Shazam in Java. : Site consulté régulièrement depuis juin 2017.
http://royvanrijn.com/blog/2010/06/creating-shazam-in-java/

[2] Wang Avery : An Industrial Strength Audio Search Algorithm. : p7-13. Ismir. 2003.

[3] Steven S. Lutz : Hokua – A Wavelet Method for Audio Fingerprinting : All Theses and
Dissertations. Brigham Young University. 2009.
