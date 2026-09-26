Team: pass the J (Team Leader: Anshu Aman)

Query:
The final Code Submission ZIP has a 50 MB upload limit, but the test set has 1,732,544 Source-1
entities against about 10M Source-2/3 records. With the train match rate (about 3.5 matches per S1), a
correct matching_results.tsv alone holds about 6M IDs (~70 MB uncompressed, ~24 MB zipped, since the
9-digit IDs barely compress). candidate_pairs.tsv must be a superset of the matches, so both files
together cannot fit in 50 MB zipped with any realistic candidate set.
Could you please clarify: (1) may the ZIP contain only matching_results.tsv, with candidate_pairs.tsv
uploaded separately or regenerated from our code; (2) or will the size limit be raised; (3) or may we
submit candidate_pairs.tsv in a compressed format (e.g. .tsv.gz / .xz) inside the ZIP?
