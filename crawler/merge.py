# Copy all files with *.Bengali --> *.Hindi
# find ./ -name '*.Bengali' -exec sh -c 'cp "$0" "${0%.Bengali}.Hindi"' {} \;

# Move all "files" in sub-folders to current folder
# find ./ -type f -print0 | xargs -0 mv -t .

import os
from pathlib import Path

f1 = open('spoken.bn','w+')
f2 = open('spoken.hi','w+')

for file in os.listdir("data/"):
    if file.endswith(".Bengali"):
        filepath_bengali = os.path.join("data/", file)
        p = Path(filepath_bengali)
        filepath_hindi = os.path.join("data/"+p.stem+".Hindi")
        if p.with_suffix('.Hindi').exists():
            lines_bengali = sum(1 for line in open(filepath_bengali))
            lines_hindi = sum(1 for line in open(filepath_hindi))
            
            if lines_bengali==lines_hindi:
                f1.write(open(filepath_bengali).read())
                f2.write(open(filepath_hindi).read())
                
f1.close()
f2.close()
