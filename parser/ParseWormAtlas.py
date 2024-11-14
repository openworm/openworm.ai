from bs4 import BeautifulSoup, NavigableString

MARKDOWN_DIR = "../processed/markdown/wormatlas"
PLAINTEXT_DIR = "../processed/plaintext/wormatlas"


class WormAtlasParser:
    markdown = None
    plaintext = None
    json = {}

    html = None

    def __init__(self, title, filename):
        ref = title.replace(" ", "_")

        self.markdown = open("%s/%s.md" % (MARKDOWN_DIR, ref), "w")

        self.markdown.write("## %s\n\n" % title)

        self.plaintext = open("%s/%s.txt" % (PLAINTEXT_DIR, ref), "w")

        self.plaintext.write("%s\n\n" % title)

        self.json["title"] = title

        with open(filename, encoding="ISO-8859-1") as f:
            self.html = f.read()

            soup = BeautifulSoup(self.html, "html.parser")

            count = 0
            for element in soup.body.find_all("table")[5].find_all("p"):
                print("%s ==================================" % count)

                if element is not None:
                    # print(table.replace('\n', '%s\n'%count))

                    print("%s -- |%s...|\n" % (count, str(element)[:200]))
                    # print(element.contents)
                    anchor = (
                        element.contents[0]["name"]
                        if (
                            len(element.contents) > 0
                            and type(element.contents[0]) is not NavigableString
                            and element.contents[0].has_attr("name")
                        )
                        else ""
                    )

                    print("Anchor: - %s" % anchor)

                    if "19" in anchor or "20" in anchor:
                        self.process_reference(element)
                    else:
                        self.process_paragraph(element)
                count += 1

        self.finalise()

    def process_reference(self, reference):
        reference.a["id"] = reference.a["name"]
        r_md = str(reference)

        print("  - REF: %s...\n" % (r_md[:80]))

        r_md = r_md.replace("\t", "  ")
        while "  " in r_md:
            r_md = r_md.replace("  ", " ")

        r_md.replace("ï¿½", "'").replace("\n", "")

        self.markdown.write("_%s_\n\n" % r_md)

        for s in reference.contents:
            print(" >>> %s" % s)
            if type(s) is NavigableString:
                self.plaintext.write("%s" % s.replace("  ", " "))
            elif s.has_attr("name"):
                pass
            elif s.has_attr("href"):
                self.plaintext.write("(%s)" % s.attrs["href"])
            else:
                for c in s.contents:
                    self.plaintext.write("%s" % c)

        self.plaintext.write("\n\n")

    def process_paragraph(self, paragraph):
        p_md = str(paragraph)

        p_md = p_md.replace("\t", "  ")
        while "  " in p_md:
            p_md = p_md.replace("  ", " ")

        p_md.replace("ï¿½", "'").replace("\n", "")

        self.markdown.write("%s\n\n" % p_md)

        for a in paragraph.find_all("a"):
            print("--- %s" % a)
            if "href" in a.attrs:
                a = a.replace_with(a.next_element)
                print("------ %s" % a)

        for em in paragraph.find_all("em"):
            print("--- %s" % em)
            cc = " ".join([str(c) for c in em.contents])
            print("---- %s" % cc)
            em = em.replace_with(cc)
            print("------ %s" % em)

        p = str(paragraph)

        p = p.replace("\t", "  ")
        while "  " in p:
            p = p.replace("  ", " ")

        p = p.replace("ï¿½", "'").replace("\n", "")

        self.plaintext.write("%s\n\n" % p[3:-4])

    def finalise(self):
        self.plaintext.close()
        self.markdown.close()
        # from bs4.diagnose import diagnose

        # diagnose(self.html)


if __name__ == "__main__":
    guides = {}

    guides["Introduction"] = "../corpus/wormatlas/Handbook - Introduction.html"
    guides["Gap Junctions"] = "../corpus/wormatlas/Handbook - Gap Junctions.html"

    with open("../processed/markdown/wormatlas/README.md", "w") as readme:
        readme.write("""
## WormAtlas Handbooks
 
The following handbooks from [WormAtlas](https://www.wormatlas.org/handbookhome.htm) have been translated to Markdown format

""")
        for g in guides:
            wbp = WormAtlasParser(g, guides[g])
            readme.write(f"**[{g}]({ g.replace(' ', '_') }.md)**\n\n")
