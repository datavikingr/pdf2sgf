# PDF2SGF

## Tsumego
I wanted a repository of Go problems that I could launch in `gnugo --mode=ascii` because I'm a terminal dork like that. I couldn't find any sgf archives of any real volume, but I could find pdf collections of these problems. 

## img2sgf
Then poking around GitHub a little bit, I found some abandonware, namely [img2sgf](https://github.com/hanysz/img2sgf), that would suit me just fine. 

However, I realized pretty quickly it could not operate headlessly, in spite of some suspicious looking sys.argv calls at the end of the script.

```python
if len(sys.argv)>3:
  sys.exit("Too many command line arguments.")

if len(sys.argv)>2:
  output_file = sys.argv[2]
else:
  output_file = None

if len(sys.argv)>1:
  input_file = sys.argv[1]
  open_file(input_file)
  if output_file == None:
    # suggest output name based on input
    output_file = os.path.splitext(input_file)[0] + ".sgf"
```
So I patched img2sgf. It's dirty (opens and closes a tkinter window for every single problem that runs through it - which is "not great"), but it definitely performs the job required and outputs .sgf files from the command line. You can find the patched version here in this repository, and the original in the link above. 

## Wrapper Script
So first I had to process the pdfs into pngs so that img2sgf could handle them, and everything after that kinda flowed from there.

Basically, we do some slicing and dicing to tighten our OCR detection windows, so we can cleanly separate the problems from each other, filter out some chaff (title pages, front matter), and then feed the problems into img2sgf en masse, one png at a time. Add a little "who has the next move?" OCR detection for exactly 3 out of 68 pdfs (* *sigh* *), and out spits a collection of tens of thousands of Go problems, which you can find in `sgfs/`. 

`pages/`, `page_slices/`, and `problems/` will be cleaned out every run. If you're intersted in those intermediary steps, clone the repo, comment out those lines, and run it. Be advised, it will take a *while*. 

## Wait, isn't this just what tasuki did for https://tsumego.tasuki.org/, but in reverse?

Yes, exactly, but with the addition of Olivier's [101books](https://101books.github.io/) of problems. The pdfs from both sources can be found in `pdfs/`. And you can find the source*code* for each of these sources here:
 - [tasuki's source](https://github.com/tasuki/tsumego-web)
 - [Olivier's source](https://github.com/101books/101books.github.io)

## The problems are out of order compared to their pdf versions
This is expected. If you'd like to specifically cross-reference which file is which problem, see `log.txt`. Use CTRL+F and start from the top, one intermediary file/step at a time and you'll find the sgf you're looking for.