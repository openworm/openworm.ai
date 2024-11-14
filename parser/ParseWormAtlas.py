from bs4 import BeautifulSoup

MARKDOWN_DIR = '../processed/markdown'
PLAINTEXT_DIR = '../processed/plaintext'

class WormBookParser:

    markdown = None
    plaintext = None
    json = {}

    html = None

    def __init__(self, filename):

        ref = filename.split('/')[-1].split('.')[0].replace(' ','')
        
        self.markdown = open('%s/%s.md'%(MARKDOWN_DIR, ref),'w')
        self.plaintext = open('%s/%s.txt'%(PLAINTEXT_DIR, ref),'w')

        with open(filename, encoding = "ISO-8859-1") as f:
            
            self.html = f.read()

            soup = BeautifulSoup(self.html, 'html.parser')

            count = 0
            for element in soup.body.find_all('table')[5].find_all('p'):
                print('%s =================================='%count)
                
                if element is not None:
                    #print(table.replace('\n', '%s\n'%count))
                    print('%s -- |%s...|\n'%(count, str(element)[:200]))
                    print(element.contents)
                    self.process_paragraph(element)
                count +=1

        self.finalise()

    def process_paragraph(self, paragraph):

        p_md =str(paragraph)
        p_md = p_md.replace('  ','').replace('ï¿½',"'")

        self.markdown.write('%s\n\n'%p_md)

        
        for a in paragraph.find_all('a'):
            print('--- %s'%a)
            if 'href' in a.attrs:
                a = a.replace_with(a.next_element)
                print('------ %s'%a)

        for em in paragraph.find_all('em'):
            print('--- %s'%em)
            cc = ' '.join([str(c) for c in em.contents])
            print('---- %s'% cc)
            em = em.replace_with(cc)
            print('------ %s'%em)

        p =str(paragraph)

        p = p.replace('  ','').replace('ï¿½',"'")

        self.plaintext.write('%s\n\n'%p[3:-4])

    def finalise(self):

        self.plaintext.close()
        self.markdown.close()
        from bs4.diagnose import diagnose

        #diagnose(self.html)


if __name__ == "__main__":

    filename = '../corpus/wormbook/Handbook - Gap Junctions.html'

    wbp = WormBookParser(filename)
