from bs4 import BeautifulSoup


class WormBookParser:

    markdown = None

    def __init__(self, filename):

        self.markdown = open('test.md','w')

        with open(filename, encoding = "ISO-8859-1") as f:
            
            html = f.read()

            soup = BeautifulSoup(html, 'html.parser')

            count = 0
            for element in soup.body.find_all('table')[5].find_all('p'):
                print('%s =================================='%count)
                #print('> ((%s))\n'%element.contents)
                
                if element is not None:
                    #print(table.replace('\n', '%s\n'%count))
                    print('%s -- |%s...|\n'%(count, str(element)[:200]))
                    #print(element.contents)
                    self.process_paragraph(str(element))
                count +=1


    def process_paragraph(self, paragraph):

        self.markdown.write('%s\n\n'%paragraph.replace('  ',''))

    def finalise(self):

        self.markdown.close()


if __name__ == "__main__":

    filename = '../corpus/wormbook/Handbook - Gap Junctions.html'

    wbp = WormBookParser(filename)
