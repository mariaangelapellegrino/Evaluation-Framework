import collections
import csv

if __name__ == "__main__":
    combined = collections.defaultdict(list)
    thefileName = "data/LP50_stats.csv"
    with  open(thefileName) as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            else:
                # SubjectID,Document1,Document2,Similarity,Time
                parts = line.split(',')
                doc1 = int(parts[1])
                doc2 = int(parts[2])
                score = int(parts[3])
                if doc1 > doc2:
                    doc1, doc2 = doc2, doc1
                combined[(doc1, doc2)].append(score)    
    
    with open('data/LP50_averageScores.csv', "wb") as csv_file:
        fieldnames = ['doc1', 'doc2', 'average']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for doc1 in range (1, 51):
            for doc2 in range (doc1 + 1, 51):

                l = combined[(doc1, doc2)]
                if len(l) == 0:
                    raise Exception()
                else:
                    averageRating = float(sum(l))/len(l)
                
                    writer.writerow({'doc1':doc1, 'doc2':doc2, 'average':averageRating})