# =============================================================================
# Najbližji par točk v ravnini
#
# Pri reševanju naloge si lahko pomagate z video lekcijami na Vimeu:
#
# * [Par najbližjih točk](http://vimeo.com/32434672)
# * [Par najbližjih točk, faza deli](http://vimeo.com/32434810)
# * [Par najbližjih točk, faza vladaj](http://vimeo.com/32435158)
# * [Par najbližjih točk, časovna zahtevnost](http://vimeo.com/32435263)
# * [Par najbližjih točk, izboljšava](http://vimeo.com/32435495)
#
# Dane imamo točke $t_1 = (x_1, y_1), \ldots, t_n = (x_n, y_n)$ v ravnini.
# Poiskati želimo dve točki $t_i$ in $t_j$, $i \neq j$, da bo razdalja med
# njima minimalna. To lahko rešimo z metodo deli in vladaj, glejte
# [opis na Wikipediji](http://en.wikipedia.org/wiki/Closest_pair_of_points_problem).
#
# V nadaljevanju naloge privzemimo, da so točke $t_1, \ldots, t_n$ shranjene
# v seznamu `t` kot pari koordinat `(x_i, y_i)`.
# =====================================================================@005826=
# 1. podnaloga
# Ker je $x \mapsto x^2$ strogo naraščajoča funkcija na $[0, \infty)$,
# lahko namesto razdalj med točkami računamo njihove kvadrate. S tem se
# izognemo računanju kvadratnih korenov. Sestavite najprej pomožno funkcijo
# `dist2(u, v)`, ki izračuna kvadrat razdalje med točkama `u` in `v`.
# Zgled:
#
#     >>> dist2((4, 1), (1, 3))
#     13
# =============================================================================
def dist2(u, v):
    return (u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2


# =====================================================================@005827=
# 2. podnaloga
# Sestavite funkcijo `uredi_po_x(t)`, ki vrne seznam `a`, v katerem so
# našteti indeksi točk iz seznama `t` glede na koordinato $x$ v naraščajočem
# redu. Se pravi, `t[a[i]]` je `i`-ta točka v `t` glede na velikost
# koordinate $x$. Zgled:
#
#     >>> uredi_po_x([(2, 3), (1, 5), (10, 7), (11, 6)])
#     [1, 0, 2, 3]
#
# Namig: Uporabite funkcijo `sorted`, ki ji podate `enumerate(t)` in
# ustrezno vrednost parametra `key`, se pravi, tako funkcijo, ki iz para
# `(i, (x, y))` izlušči `x`.
# =============================================================================
def uredi_po_x(t):
    pari = [(i, t[i]) for i in range(len(t))]
    pari = sorted(pari, key=lambda indeks: indeks[1][0])
    return [i[0] for i in pari]


# =====================================================================@005828=
# 3. podnaloga
# Sestavite funkcijo `uredi_po_y(t)`, ki vrne seznam `a`, v katerem so
# našteti indeksi točk iz seznama `t` glede na koordinato $y$ v naraščajočem
# redu. Se pravi, `t[a[i]]` je `i`-ta točka v `t` glede na velikost
# koordinate $y$. Zgled:
#
#     >>> uredi_po_y([(2, 3), (1, 5), (10, 7), (11, 6)])
#     [0, 1, 3, 2]
#
# Namig: Uporabite funkcijo `sorted`, ki ji podate `enumerate(t)` in
# ustrezno vrednost parametra `key`, se pravi, tako funkcijo, ki iz para
# `(i, (x, y))` izlušči `y`.
# =============================================================================
# def uredi_po_y(t):
# return [i for (i, u) in sorted(enumerate(t), key=lambda indeks: indeks[1][1])]


def uredi_po_y(t):
    return [i for (i, u) in sorted(enumerate(t), key=lambda p: p[1][1])]


# =====================================================================@005829=
# 4. podnaloga
# Sestavite funkcijo `najblizji(t)`, ki sprejme seznam točk v ravnini in
# vrne _indeksa_ `(i, j)`, $i < j$, tistih dveh točk `t[i]` in `t[j]`, ki
# sta si najbližji. Če je seznam `t` prazen ali vsebuje samo eno točko,
# naj funkcija vrne `None`. Zgled:
#
#     >>> najblizji([(1, 2), (10, 20), (3, 4)])
#     (0, 2)
#     >>> najblizji([(1, 2)])
#     None
#
# Funkcija mora delovati v času $O(n \log n)$ ali vsaj $O(n \log^2 n)$.
# Nasveti za reševanje:
#
# * V funkciji `najblizji` najprej naredite seznama `ix` in `iy`, ki naj
#   vsebujeta indekse točk iz `t`, ki so urejeni glede na koordinati $x$
#   in $y$.
# * Nato znotraj funkcije `najblizji` definirajte pomožno funkcijo
#   `isci(i, j)`, ki poišče najbližji par točk med tistimi z indeksi
#   iz `ix[i:j+1]`. Ta funkcija je rekurzivna.
# * V funkciji `isci` najprej poskrbite za robne primere.
# * Nato naredite fazo "deli", ki točke glede na koordinato $x$ razdeli
#   na dva dela in rekurzivno poišče rešitev v vsakem delu posebej.
# * Sledi faza "vladaj". Med obema odgovoroma izberite par točk, ki je
#   bližji in izračunajte kvadrat razdalje med točkama.
# * Izračunajte $x$ koordinato delilne črte.
# * Poiščite indekse točk, ki ležijo v pasu.
# * Preiščite pas, saj ta lahko vsebuje par točk, ki sta si bližje kot
#   najbližji par v levem oz. desnem delu.
# =============================================================================
def najblizji(t):
    ix = uredi_po_x(t)
    iy = uredi_po_y(t)

    def isci(i, j):
        if i == j:
            return (i, i + 1)
        if i == j - 1:
            if dist2(ix[i], ix[i + 1]) > dist2(ix[i], ix[i + 2]):
                return (i, i + 1)
            else:
                return (i, i + 2)
        else:
            deli = j // 2
            kandidat1 = isci(i, deli)
            kandidat2 = isci(deli, j)
            if dist2(kandidat1) < dist2(kandidat2):
                boljsi = kandidat1
            else:
                boljsi = kandidat2
            razdalja = dist2(boljsi)
            # še pas
            return boljsi


# ============================================================================@

"Če vam Python sporoča, da je v tej vrstici sintaktična napaka,"
"se napaka v resnici skriva v zadnjih vrsticah vaše kode."

"Kode od tu naprej NE SPREMINJAJTE!"


import json
import os
import re
import shutil
import sys
import traceback
import urllib.error
import urllib.request

import io
import sys
from contextlib import contextmanager


class VisibleStringIO(io.StringIO):
    def read(self, size=None):
        x = io.StringIO.read(self, size)
        print(x, end="")
        return x

    def readline(self, size=None):
        line = io.StringIO.readline(self, size)
        print(line, end="")
        return line


class Check:
    parts = None
    current_part = None
    part_counter = None

    @staticmethod
    def has_solution(part):
        return part["solution"].strip() != ""

    @staticmethod
    def initialize(parts):
        Check.parts = parts
        for part in Check.parts:
            part["valid"] = True
            part["feedback"] = []
            part["secret"] = []

    @staticmethod
    def part():
        if Check.part_counter is None:
            Check.part_counter = 0
        else:
            Check.part_counter += 1
        Check.current_part = Check.parts[Check.part_counter]
        return Check.has_solution(Check.current_part)

    @staticmethod
    def feedback(message, *args, **kwargs):
        Check.current_part["feedback"].append(message.format(*args, **kwargs))

    @staticmethod
    def error(message, *args, **kwargs):
        Check.current_part["valid"] = False
        Check.feedback(message, *args, **kwargs)

    @staticmethod
    def clean(x, digits=6, typed=False):
        t = type(x)
        if t is float:
            x = round(x, digits)
            # Since -0.0 differs from 0.0 even after rounding,
            # we change it to 0.0 abusing the fact it behaves as False.
            v = x if x else 0.0
        elif t is complex:
            v = complex(
                Check.clean(x.real, digits, typed), Check.clean(x.imag, digits, typed)
            )
        elif t is list:
            v = list([Check.clean(y, digits, typed) for y in x])
        elif t is tuple:
            v = tuple([Check.clean(y, digits, typed) for y in x])
        elif t is dict:
            v = sorted(
                [
                    (Check.clean(k, digits, typed), Check.clean(v, digits, typed))
                    for (k, v) in x.items()
                ]
            )
        elif t is set:
            v = sorted([Check.clean(y, digits, typed) for y in x])
        else:
            v = x
        return (t, v) if typed else v

    @staticmethod
    def secret(x, hint=None, clean=None):
        clean = Check.get("clean", clean)
        Check.current_part["secret"].append((str(clean(x)), hint))

    @staticmethod
    def equal(expression, expected_result, clean=None, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        actual_result = eval(expression, global_env)
        if clean(actual_result) != clean(expected_result):
            Check.error(
                "Izraz {0} vrne {1!r} namesto {2!r}.",
                expression,
                actual_result,
                expected_result,
            )
            return False
        else:
            return True

    @staticmethod
    def approx(expression, expected_result, tol=1e-6, env=None, update_env=None):
        try:
            import numpy as np
        except ImportError:
            Check.error("Namestiti morate numpy.")
            return False
        if not isinstance(expected_result, np.ndarray):
            Check.error("Ta funkcija je namenjena testiranju za tip np.ndarray.")

        if env is None:
            env = dict()
        env.update({"np": np})
        global_env = Check.init_environment(env=env, update_env=update_env)
        actual_result = eval(expression, global_env)
        if type(actual_result) is not type(expected_result):
            Check.error(
                "Rezultat ima napačen tip. Pričakovan tip: {}, dobljen tip: {}.",
                type(expected_result).__name__,
                type(actual_result).__name__,
            )
            return False
        exp_shape = expected_result.shape
        act_shape = actual_result.shape
        if exp_shape != act_shape:
            Check.error(
                "Obliki se ne ujemata. Pričakovana oblika: {}, dobljena oblika: {}.",
                exp_shape,
                act_shape,
            )
            return False
        try:
            np.testing.assert_allclose(
                expected_result, actual_result, atol=tol, rtol=tol
            )
            return True
        except AssertionError as e:
            Check.error("Rezultat ni pravilen." + str(e))
            return False

    @staticmethod
    def run(statements, expected_state, clean=None, env=None, update_env=None):
        code = "\n".join(statements)
        statements = "  >>> " + "\n  >>> ".join(statements)
        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        exec(code, global_env)
        errors = []
        for (x, v) in expected_state.items():
            if x not in global_env:
                errors.append(
                    "morajo nastaviti spremenljivko {0}, vendar je ne".format(x)
                )
            elif clean(global_env[x]) != clean(v):
                errors.append(
                    "nastavijo {0} na {1!r} namesto na {2!r}".format(
                        x, global_env[x], v
                    )
                )
        if errors:
            Check.error("Ukazi\n{0}\n{1}.", statements, ";\n".join(errors))
            return False
        else:
            return True

    @staticmethod
    @contextmanager
    def in_file(filename, content, encoding=None):
        encoding = Check.get("encoding", encoding)
        with open(filename, "w", encoding=encoding) as f:
            for line in content:
                print(line, file=f)
        old_feedback = Check.current_part["feedback"][:]
        yield
        new_feedback = Check.current_part["feedback"][len(old_feedback) :]
        Check.current_part["feedback"] = old_feedback
        if new_feedback:
            new_feedback = ["\n    ".join(error.split("\n")) for error in new_feedback]
            Check.error(
                "Pri vhodni datoteki {0} z vsebino\n  {1}\nso se pojavile naslednje"
                " napake:\n- {2}",
                filename,
                "\n  ".join(content),
                "\n- ".join(new_feedback),
            )

    @staticmethod
    @contextmanager
    def input(content, visible=None):
        old_stdin = sys.stdin
        old_feedback = Check.current_part["feedback"][:]
        try:
            with Check.set_stringio(visible):
                sys.stdin = Check.get("stringio")("\n".join(content) + "\n")
                yield
        finally:
            sys.stdin = old_stdin
        new_feedback = Check.current_part["feedback"][len(old_feedback) :]
        Check.current_part["feedback"] = old_feedback
        if new_feedback:
            new_feedback = ["\n  ".join(error.split("\n")) for error in new_feedback]
            Check.error(
                "Pri vhodu\n  {0}\nso se pojavile naslednje napake:\n- {1}",
                "\n  ".join(content),
                "\n- ".join(new_feedback),
            )

    @staticmethod
    def out_file(filename, content, encoding=None):
        encoding = Check.get("encoding", encoding)
        with open(filename, encoding=encoding) as f:
            out_lines = f.readlines()
        equal, diff, line_width = Check.difflines(out_lines, content)
        if equal:
            return True
        else:
            Check.error(
                "Izhodna datoteka {0}\n  je enaka{1}  namesto:\n  {2}",
                filename,
                (line_width - 7) * " ",
                "\n  ".join(diff),
            )
            return False

    @staticmethod
    def output(expression, content, env=None, update_env=None):
        global_env = Check.init_environment(env=env, update_env=update_env)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        too_many_read_requests = False
        try:
            exec(expression, global_env)
        except EOFError:
            too_many_read_requests = True
        finally:
            output = sys.stdout.getvalue().rstrip().splitlines()
            sys.stdout = old_stdout
        equal, diff, line_width = Check.difflines(output, content)
        if equal and not too_many_read_requests:
            return True
        else:
            if too_many_read_requests:
                Check.error("Program prevečkrat zahteva uporabnikov vnos.")
            if not equal:
                Check.error(
                    "Program izpiše{0}  namesto:\n  {1}",
                    (line_width - 13) * " ",
                    "\n  ".join(diff),
                )
            return False

    @staticmethod
    def difflines(actual_lines, expected_lines):
        actual_len, expected_len = len(actual_lines), len(expected_lines)
        if actual_len < expected_len:
            actual_lines += (expected_len - actual_len) * ["\n"]
        else:
            expected_lines += (actual_len - expected_len) * ["\n"]
        equal = True
        line_width = max(
            len(actual_line.rstrip())
            for actual_line in actual_lines + ["Program izpiše"]
        )
        diff = []
        for out, given in zip(actual_lines, expected_lines):
            out, given = out.rstrip(), given.rstrip()
            if out != given:
                equal = False
            diff.append(
                "{0} {1} {2}".format(
                    out.ljust(line_width), "|" if out == given else "*", given
                )
            )
        return equal, diff, line_width

    @staticmethod
    def init_environment(env=None, update_env=None):
        global_env = globals()
        if not Check.get("update_env", update_env):
            global_env = dict(global_env)
        global_env.update(Check.get("env", env))
        return global_env

    @staticmethod
    def generator(
        expression,
        expected_values,
        should_stop=None,
        further_iter=None,
        clean=None,
        env=None,
        update_env=None,
    ):
        from types import GeneratorType

        global_env = Check.init_environment(env=env, update_env=update_env)
        clean = Check.get("clean", clean)
        gen = eval(expression, global_env)
        if not isinstance(gen, GeneratorType):
            Check.error("Izraz {0} ni generator.", expression)
            return False

        try:
            for iteration, expected_value in enumerate(expected_values):
                actual_value = next(gen)
                if clean(actual_value) != clean(expected_value):
                    Check.error(
                        "Vrednost #{0}, ki jo vrne generator {1} je {2!r} namesto"
                        " {3!r}.",
                        iteration,
                        expression,
                        actual_value,
                        expected_value,
                    )
                    return False
            for _ in range(Check.get("further_iter", further_iter)):
                next(gen)  # we will not validate it
        except StopIteration:
            Check.error("Generator {0} se prehitro izteče.", expression)
            return False

        if Check.get("should_stop", should_stop):
            try:
                next(gen)
                Check.error("Generator {0} se ne izteče (dovolj zgodaj).", expression)
            except StopIteration:
                pass  # this is fine
        return True

    @staticmethod
    def summarize():
        for i, part in enumerate(Check.parts):
            if not Check.has_solution(part):
                print("{0}. podnaloga je brez rešitve.".format(i + 1))
            elif not part["valid"]:
                print("{0}. podnaloga nima veljavne rešitve.".format(i + 1))
            else:
                print("{0}. podnaloga ima veljavno rešitev.".format(i + 1))
            for message in part["feedback"]:
                print("  - {0}".format("\n    ".join(message.splitlines())))

    settings_stack = [
        {
            "clean": clean.__func__,
            "encoding": None,
            "env": {},
            "further_iter": 0,
            "should_stop": False,
            "stringio": VisibleStringIO,
            "update_env": False,
        }
    ]

    @staticmethod
    def get(key, value=None):
        if value is None:
            return Check.settings_stack[-1][key]
        return value

    @staticmethod
    @contextmanager
    def set(**kwargs):
        settings = dict(Check.settings_stack[-1])
        settings.update(kwargs)
        Check.settings_stack.append(settings)
        try:
            yield
        finally:
            Check.settings_stack.pop()

    @staticmethod
    @contextmanager
    def set_clean(clean=None, **kwargs):
        clean = clean or Check.clean
        with Check.set(clean=(lambda x: clean(x, **kwargs)) if kwargs else clean):
            yield

    @staticmethod
    @contextmanager
    def set_environment(**kwargs):
        env = dict(Check.get("env"))
        env.update(kwargs)
        with Check.set(env=env):
            yield

    @staticmethod
    @contextmanager
    def set_stringio(stringio):
        if stringio is True:
            stringio = VisibleStringIO
        elif stringio is False:
            stringio = io.StringIO
        if stringio is None or stringio is Check.get("stringio"):
            yield
        else:
            with Check.set(stringio=stringio):
                yield


def _validate_current_file():
    def extract_parts(filename):
        with open(filename, encoding="utf-8") as f:
            source = f.read()
        part_regex = re.compile(
            r"# =+@(?P<part>\d+)=\s*\n"  # beginning of header
            r"(\s*#( [^\n]*)?\n)+?"  # description
            r"\s*# =+\s*?\n"  # end of header
            r"(?P<solution>.*?)"  # solution
            r"(?=\n\s*# =+@)",  # beginning of next part
            flags=re.DOTALL | re.MULTILINE,
        )
        parts = [
            {"part": int(match.group("part")), "solution": match.group("solution")}
            for match in part_regex.finditer(source)
        ]
        # The last solution extends all the way to the validation code,
        # so we strip any trailing whitespace from it.
        parts[-1]["solution"] = parts[-1]["solution"].rstrip()
        return parts

    def backup(filename):
        backup_filename = None
        suffix = 1
        while not backup_filename or os.path.exists(backup_filename):
            backup_filename = "{0}.{1}".format(filename, suffix)
            suffix += 1
        shutil.copy(filename, backup_filename)
        return backup_filename

    def submit_parts(parts, url, token):
        submitted_parts = []
        for part in parts:
            if Check.has_solution(part):
                submitted_part = {
                    "part": part["part"],
                    "solution": part["solution"],
                    "valid": part["valid"],
                    "secret": [x for (x, _) in part["secret"]],
                    "feedback": json.dumps(part["feedback"]),
                }
                if "token" in part:
                    submitted_part["token"] = part["token"]
                submitted_parts.append(submitted_part)
        data = json.dumps(submitted_parts).encode("utf-8")
        headers = {"Authorization": token, "content-type": "application/json"}
        request = urllib.request.Request(url, data=data, headers=headers)
        # This is a workaround because some clients (and not macOS ones!) report
        # <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1129)>
        import ssl

        context = ssl._create_unverified_context()
        response = urllib.request.urlopen(request, context=context)
        # When the issue is resolved, the following should be used
        # response = urllib.request.urlopen(request)
        return json.loads(response.read().decode("utf-8"))

    def update_attempts(old_parts, response):
        updates = {}
        for part in response["attempts"]:
            part["feedback"] = json.loads(part["feedback"])
            updates[part["part"]] = part
        for part in old_parts:
            valid_before = part["valid"]
            part.update(updates.get(part["part"], {}))
            valid_after = part["valid"]
            if valid_before and not valid_after:
                wrong_index = response["wrong_indices"].get(str(part["part"]))
                if wrong_index is not None:
                    hint = part["secret"][wrong_index][1]
                    if hint:
                        part["feedback"].append("Namig: {}".format(hint))

    filename = os.path.abspath(sys.argv[0])
    file_parts = extract_parts(filename)
    Check.initialize(file_parts)

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1ODI2LCJ1c2VyIjo2OTI3fQ:1pMB8g:Gfi2_0qyxiGRhqtSmQkFwnvXj4fxH0IwYTB-bHcD9C0"
        try:
            test_data = [
                ("""dist2((4, 1), (1, 3))""", 13),
                ("""dist2((-4, 1), (1, 3))""", 29),
                ("""dist2((4, 1), (1, -3))""", 25),
                ("""dist2((4, 1), (6, -3))""", 20),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
        except:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1ODI3LCJ1c2VyIjo2OTI3fQ:1pMB8g:Mp798mo0b-4UiJNNY__GUkTXfcCakhBSiT16n4BK9r4"
        try:
            Check.equal("""uredi_po_x([(2,3), (1,5), (10,7), (11,6)])""", [1, 0, 2, 3])
            Check.secret(
                uredi_po_x(
                    [
                        ((30 * i - 5 * i * i) % 120, (27 * i + 8 * i * i) % 100)
                        for i in range(1000)
                    ]
                )
            )
        except:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1ODI4LCJ1c2VyIjo2OTI3fQ:1pMB8g:NK0956k8mGiJsNmC1t-kSD4Evo_wjY_gEPiqqnO0eeY"
        try:
            Check.equal("""uredi_po_y([(2,3), (1,5), (10,7), (11,6)])""", [0, 1, 3, 2])
            Check.secret(
                uredi_po_y(
                    [
                        ((30 * i - 5 * i * i) % 120, (27 * i + 8 * i * i) % 100)
                        for i in range(1000)
                    ]
                )
            )
        except:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1ODI5LCJ1c2VyIjo2OTI3fQ:1pMB8g:a4RFMPqRZFd2kE9kTZ3Yvii74S63dwHS9qFc50iX3B4"
        try:
            Check.equal("""najblizji([])""", None)
            Check.equal("""najblizji([(1,2)])""", None)
            Check.equal("""najblizji([(1,2), (3,4)])""", (0, 1))
            Check.equal("""najblizji([(1,2), (3,4), (10, 20)])""", (0, 1))
            Check.equal("""najblizji([(1,2), (10, 20), (3,4)])""", (0, 2))
            Check.equal("""najblizji([(i*i, 0) for i in range(0,10)])""", (0, 1))
            Check.equal("""najblizji([(0, i*i) for i in range(0,10)])""", (0, 1))
            Check.equal(
                """najblizji([(i**3, i) for i in range(-10,10) if i != 1])""", (9, 10)
            )

            Check.secret([(i * i * i, j * j) for i in range(50) for j in range(50)])
            Check.secret(
                [(i * i * i, j * j) for i in range(-50, 50) for j in range(-50, 50)]
            )
        except:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    print("Shranjujem rešitve na strežnik... ", end="")
    try:
        url = "https://www.projekt-tomo.si/api/attempts/submit/"
        token = "Token e08703d9c33e76282eef8a8d6a07def9e7a423aa"
        response = submit_parts(Check.parts, url, token)
    except urllib.error.URLError:
        message = (
            "\n"
            "-------------------------------------------------------------------\n"
            "PRI SHRANJEVANJU JE PRIŠLO DO NAPAKE!\n"
            "Preberite napako in poskusite znova ali se posvetujte z asistentom.\n"
            "-------------------------------------------------------------------\n"
        )
        print(message)
        traceback.print_exc()
        print(message)
        sys.exit(1)
    else:
        print("Rešitve so shranjene.")
        update_attempts(Check.parts, response)
        if "update" in response:
            print("Updating file... ", end="")
            backup_filename = backup(filename)
            with open(__file__, "w", encoding="utf-8") as f:
                f.write(response["update"])
            print("Previous file has been renamed to {0}.".format(backup_filename))
            print("If the file did not refresh in your editor, close and reopen it.")
    Check.summarize()


if __name__ == "__main__":
    _validate_current_file()
