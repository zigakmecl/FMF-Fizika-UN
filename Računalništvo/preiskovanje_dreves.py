# =============================================================================
# Preiskovanje dreves
#
# Dan je razred `Drevo`, ki predstavlja dvojiško drevo. Konstruktor je že
# implementiran; prav tako metodi `__repr__` in `__eq__`. Drevo na spodnji
# (ASCII art) sliki:
#
#          5
#        /   \
#       3     2
#      /     / \
#     1     6   9
#
# sestavimo takole:
#
#     >>> d = Drevo(5,
#                   levo=Drevo(3, levo=Drevo(1)),
#                   desno=Drevo(2, levo=Drevo(6), desno=Drevo(9)))
# =====================================================================@005912=
# 1. podnaloga
# Razredu dodajte metodo `vsota(self)`, ki vrne vsoto vseh števil v
# drevesu. Zgled (kjer je `d` kot zgoraj):
#
#     >>> d.vsota()
#     26
# =============================================================================
class Drevo:
    def __init__(self, *args, **kwargs):
        if args:
            self.prazno = False
            self.vsebina = args[0]
            self.levo = kwargs.get("levo", Drevo())
            self.desno = kwargs.get("desno", Drevo())
        else:
            self.prazno = True

    def __repr__(self, zamik=""):
        if self.prazno:
            return "Drevo()".format(zamik)
        elif self.levo.prazno and self.desno.prazno:
            return "Drevo({1})".format(zamik, self.vsebina)
        else:
            return "Drevo({1},\n{0}      levo = {2},\n{0}      desno = {3})".format(
                zamik,
                self.vsebina,
                self.levo.__repr__(zamik + "             "),
                self.desno.__repr__(zamik + "              "),
            )

    def __eq__(self, other):
        return (self.prazno and other.prazno) or (
            not self.prazno
            and not other.prazno
            and self.vsebina == other.vsebina
            and self.levo == other.levo
            and self.desno == other.desno
        )

    def vsota(self):
        if self.prazno:
            return 0
        else:
            return self.vsebina + self.desno.vsota() + self.levo.vsota()


# =====================================================================@005913=
# 2. podnaloga
# Dodajte metodo `stevilo_listov(self)`, ki vrne število listov v drevesu.
# Zgled (kjer je `d` kot zgoraj):
#
#     >>> d.stevilo_listov()
#     3
# =============================================================================
class Drevo(Drevo):
    def stevilo_listov(self):
        if self.prazno:
            return 0
        l = self.levo
        d = self.desno
        # if self.levo.self.prazno and self.levo.self.prazno:
        if l.prazno and d.prazno:
            return 1
        else:
            return self.desno.stevilo_listov() + self.levo.stevilo_listov()


# =====================================================================@005914=
# 3. podnaloga
# Dodajte metodo `minimum(self)`, ki vrne najmanjše število v drevesu.
# Če je drevo prazno, naj metoda vrne `None`. Zgled (kjer je `d` kot
# zgoraj):
#
#     >>> d.minimum()
#     1
#     >>> Drevo().minimum()
#     None
# =============================================================================
class Drevo(Drevo):
    def minimum(self):
        sez = []
        if self.prazno:
            return None

        l = self.levo
        d = self.desno

        if l.prazno and d.prazno:
            return self.vsebina
        else:
            sez.append(self.vsebina)

        if not l.prazno:
            sez.append(self.levo.minimum())
            return min(sez)
        if not d.prazno:
            sez.append(self.desno.minimum())
            return min(sez)


# =====================================================================@005915=
# 4. podnaloga
# Sestavite metodo `premi_pregled(self)`, ki vrne generator, ki vrača
# _vsebino_ vozlišč drevesa v _premem vrstnem redu_ (pre-order). To pomeni,
# da najprej obiščemo koren drevesa, nato levo poddrevo in na koncu še
# desno poddrevo. Vozlišča poddreves obiskujemo po enakem pravilu. Zgled:
#
#     >>> [x for x in d.premi_pregled()]
#     [5, 3, 1, 2, 6, 9]
#
# Opomba: Za več podrobnosti o pregledovanju dreves si lahko ogledate
# članek [Tree traversal](http://en.wikipedia.org/wiki/Tree_traversal)
# na Wikipediji.
# =============================================================================
class Drevo(Drevo):
    def premi_pregled(self):
        if self.prazno:
            return []
        l = self.levo
        d = self.desno

        if l.prazno and d.prazno:
            return [self.vsebina]

        if not self.prazno:
            gl = [self.vsebina]
            gl.extend([i for i in self.levo.premi_pregled()])

            pom = [j for j in self.desno.premi_pregled()]
            gl.extend(pom)
            return gl


# =====================================================================@005916=
# 5. podnaloga
# Sestavite metodo `vmesni_pregled(self)`, ki vrne generator, ki vrača
# _vsebino_ vozlišč drevesa v _vmesnem vrstnem redu_ (in-order). To pomeni,
# da najprej obiščemo levo poddrevo, nato koren drevesa in na koncu še
# desno poddrevo. Vozlišča poddreves obiskujemo po enakem pravilu. Zgled:
#
#     >>> [x for x in d.vmesni_pregled()]
#     [1, 3, 5, 6, 2, 9]
# =============================================================================
"""
class Drevo(Drevo):
    def vmesni_pregled(self):
        if self.prazno:
            return []
        l = self.levo
        d = self.desno

        if l.prazno and d.prazno:
            return [self.vsebina]

        if not self.prazno:
            gl = [i for i in self.levo.vmesni_pregled()]
            gl.extend([self.vsebina])
            
            pom = [j for j in self.desno.vmesni_pregled()]
            gl.extend(pom)
            
            return gl
"""


class Drevo(Drevo):
    def vmesni_pregled(self):
        if not self.prazno:
            for x in self.levo.vmesni_pregled():
                yield x
            yield self.vsebina
            for x in self.desno.vmesni_pregled():
                yield x


# =====================================================================@005917=
# 6. podnaloga
# Sestavite metodo `po_nivojih(self)`, ki vrne generator, ki vrača
# _vsebino_ vozlišč drevesa _po nivojih_ (level-order). To pomeni, da
# najprej obiščemo koren, nato vsa vozlišča, ki so na globini 1, nato
# vsa vozlišča, ki so na globini 2 itn. Vsa vozlišča na isti globini
# naštejemo od leve proti desni. Zgled:
#
#     >>> [x for x in d.po_nivojih()]
#     [5, 3, 2, 1, 6, 9]
# =============================================================================
class Drevo(Drevo):
    def po_nivojih(self):
        if self.prazno:
            return []

        # sez = [self.vsebina]
        sez2 = [self]
        dolz = len(self.vmesni_pregled())
        while len(sez2) < dolz:
            for i in sez2:
                if not self.prazno:
                    l = i.levo
                    d = i.desno
                    if not l.prazno:
                        sez2.append(l)
                    if not d.prazno:
                        sez2.append(d)

        return [i.vsebina for i in sez2]


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
                "Pri vhodni datoteki {0} z vsebino\n  {1}\nso se pojavile naslednje napake:\n- {2}",
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
                        "Vrednost #{0}, ki jo vrne generator {1} je {2!r} namesto {3!r}.",
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
        ] = "eyJwYXJ0Ijo1OTEyLCJ1c2VyIjo2OTI3fQ:1pDXt8:_6uNdSi6AXtyR0EjZJEf8erBzLJH_iJH44fQWLkSjno"
        try:
            test_data = [
                ("Drevo().vsota()", 0),
                (
                    "Drevo(5, levo=Drevo(3, levo=Drevo(1)), desno=Drevo(2, levo=Drevo(6), desno=Drevo(9))).vsota()",
                    26,
                ),
                ("Drevo(3).vsota()", 3),
                (
                    "Drevo(5, levo=Drevo(4, desno=Drevo(2)), desno=Drevo(3, levo=Drevo(4), desno=Drevo(4))).vsota()",
                    22,
                ),
                (
                    "Drevo(5, levo=Drevo(4, desno=Drevo(2)), desno=Drevo(3, levo=Drevo(5, levo=Drevo(3, levo=Drevo(1)), desno=Drevo(2, levo=Drevo(6), desno=Drevo(9))), desno=Drevo(3, levo=Drevo(4), desno=Drevo(4)))).vsota()",
                    51,
                ),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
            _drevesa = [Drevo(), Drevo()]
            for i in range(1, 20):
                _drevesa.append(Drevo(i, levo=_drevesa[-1], desno=_drevesa[-2]))
                Check.secret(_drevesa[-1].vsota())
        except:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1OTEzLCJ1c2VyIjo2OTI3fQ:1pDXt8:vNu1fk8tnwX-KLeNnXJwtc7i_FloR90j-ZvDU4dY-_w"
        try:
            test_data = [
                (
                    "Drevo(5, levo=Drevo(3, levo=Drevo(1)), desno=Drevo(2, levo=Drevo(6), desno=Drevo(9))).stevilo_listov()",
                    3,
                ),
                ("Drevo().stevilo_listov()", 0),
                ("Drevo(3).stevilo_listov()", 1),
                (
                    "Drevo(5, levo=Drevo(4, desno=Drevo(2)), desno=Drevo(3, levo=Drevo(4), desno=Drevo(4))).stevilo_listov()",
                    3,
                ),
                (
                    "Drevo(5, levo=Drevo(4, desno=Drevo(2)), desno=Drevo(3, levo=Drevo(5, levo=Drevo(3, levo=Drevo(1)), desno=Drevo(2, levo=Drevo(6), desno=Drevo(9))), desno=Drevo(3, levo=Drevo(4), desno=Drevo(4)))).stevilo_listov()",
                    6,
                ),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
            _drevesa = [Drevo(), Drevo()]
            for i in range(1, 20):
                _drevesa.append(Drevo(i, levo=_drevesa[-1], desno=_drevesa[-2]))
                Check.secret(_drevesa[-1].stevilo_listov())
        except:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1OTE0LCJ1c2VyIjo2OTI3fQ:1pDXt8:AA4pQgm0La6Qwz_JuwyCSgto4nP_aLM3lMfxqW2gf-s"
        try:
            test_data = [
                (
                    "Drevo(5, levo=Drevo(3, levo=Drevo(1)), desno=Drevo(2, levo=Drevo(6), desno=Drevo(9))).minimum()",
                    1,
                ),
                ("Drevo().minimum()", None),
                ("Drevo(3).minimum()", 3),
                (
                    "Drevo(5, levo=Drevo(4, desno=Drevo(2)), desno=Drevo(3, levo=Drevo(4), desno=Drevo(4))).minimum()",
                    2,
                ),
                (
                    "Drevo(5, levo=Drevo(4, desno=Drevo(21)), desno=Drevo(23, levo=Drevo(5, levo=Drevo(13, levo=Drevo(11)), desno=Drevo(24, levo=Drevo(6), desno=Drevo(9))), desno=Drevo(13, levo=Drevo(4), desno=Drevo(4)))).minimum()",
                    4,
                ),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
            _drevesa = [Drevo(), Drevo()]
            for i in range(1, 20):
                _drevesa.append(Drevo(i, levo=_drevesa[-1], desno=_drevesa[-2]))
                Check.secret(_drevesa[-1].minimum())
            _drevesa = [Drevo(), Drevo()]
            for i in range(1, 20):
                _drevesa.append(Drevo(-i, levo=_drevesa[-1], desno=_drevesa[-2]))
                Check.secret(_drevesa[-1].minimum())
        except:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1OTE1LCJ1c2VyIjo2OTI3fQ:1pDXt8:JtagtFE5yZj0fLt5I-4eBkpOMAuFjrAlG58Avid-Fuo"
        try:
            test_data = [
                (
                    "list(Drevo(5, levo=Drevo(3, levo=Drevo(1)), desno=Drevo(2, levo=Drevo(6), desno=Drevo(9))).premi_pregled())",
                    [5, 3, 1, 2, 6, 9],
                ),
                ("list(Drevo().premi_pregled())", []),
                ("list(Drevo(3).premi_pregled())", [3]),
                (
                    "list(Drevo(5, levo=Drevo(4, desno=Drevo(2)), desno=Drevo(3)).premi_pregled())",
                    [5, 4, 2, 3],
                ),
                (
                    "list(Drevo(5, levo=Drevo(4, desno=Drevo(2)), desno=Drevo(3, levo=Drevo(4), desno=Drevo(4))).premi_pregled())",
                    [5, 4, 2, 3, 4, 4],
                ),
                (
                    "list(Drevo(-2, levo=Drevo(4, desno=Drevo(21)), desno=Drevo(23, levo=Drevo(5, levo=Drevo(13, levo=Drevo(11)), desno=Drevo(24, levo=Drevo(6), desno=Drevo(9))), desno=Drevo(73, levo=Drevo(44), desno=Drevo(54)))).premi_pregled())",
                    [-2, 4, 21, 23, 5, 13, 11, 24, 6, 9, 73, 44, 54],
                ),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
            _drevesa = [Drevo(), Drevo()]
            for i in range(1, 12):
                _drevesa.append(Drevo(i, levo=_drevesa[-1], desno=_drevesa[-2]))
                Check.secret(list(_drevesa[-1].premi_pregled()))
        except:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1OTE2LCJ1c2VyIjo2OTI3fQ:1pDXt8:pUWY2pq0X8jT-Yra35y9ChZ3Cj8b-JcnH0iJG-xx8hs"
        try:
            test_data = [
                (
                    "list(Drevo(5, levo=Drevo(3, levo=Drevo(1)), desno=Drevo(2, levo=Drevo(6), desno=Drevo(9))).vmesni_pregled())",
                    [1, 3, 5, 6, 2, 9],
                ),
                ("list(Drevo().vmesni_pregled())", []),
                ("list(Drevo(3).vmesni_pregled())", [3]),
                (
                    "list(Drevo(5, levo=Drevo(4, desno=Drevo(2)), desno=Drevo(3)).vmesni_pregled())",
                    [4, 2, 5, 3],
                ),
                (
                    "list(Drevo(5, levo=Drevo(4, desno=Drevo(2)), desno=Drevo(3, levo=Drevo(4), desno=Drevo(4))).vmesni_pregled())",
                    [4, 2, 5, 4, 3, 4],
                ),
                (
                    "list(Drevo(-2, levo=Drevo(4, desno=Drevo(21)), desno=Drevo(23, levo=Drevo(5, levo=Drevo(13, levo=Drevo(11)), desno=Drevo(24, levo=Drevo(6), desno=Drevo(9))), desno=Drevo(73, levo=Drevo(44), desno=Drevo(54)))).vmesni_pregled())",
                    [4, 21, -2, 11, 13, 5, 6, 24, 9, 23, 44, 73, 54],
                ),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
            _drevesa = [Drevo(), Drevo()]
            for i in range(1, 12):
                _drevesa.append(Drevo(i, levo=_drevesa[-1], desno=_drevesa[-2]))
                Check.secret(list(_drevesa[-1].vmesni_pregled()))
        except:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1OTE3LCJ1c2VyIjo2OTI3fQ:1pDXt8:1yMQ4lJU9Jqhj_bPmZzoddR0MThbQbiXURnMXwjYTMw"
        try:
            test_data = [
                (
                    "list(Drevo(5, levo=Drevo(3, levo=Drevo(1)), desno=Drevo(2, levo=Drevo(6), desno=Drevo(9))).po_nivojih())",
                    [5, 3, 2, 1, 6, 9],
                ),
                ("list(Drevo().po_nivojih())", []),
                ("list(Drevo(3).po_nivojih())", [3]),
                (
                    "list(Drevo(5, levo=Drevo(4, desno=Drevo(2)), desno=Drevo(3)).po_nivojih())",
                    [5, 4, 3, 2],
                ),
                (
                    "list(Drevo(5, levo=Drevo(4, desno=Drevo(2)), desno=Drevo(3, levo=Drevo(4), desno=Drevo(4))).po_nivojih())",
                    [5, 4, 3, 2, 4, 4],
                ),
                (
                    "list(Drevo(-2, levo=Drevo(4, desno=Drevo(21)), desno=Drevo(23, levo=Drevo(5, levo=Drevo(13, levo=Drevo(11)), desno=Drevo(24, levo=Drevo(6), desno=Drevo(9))), desno=Drevo(73, levo=Drevo(44), desno=Drevo(54)))).po_nivojih())",
                    [-2, 4, 23, 21, 5, 73, 13, 24, 44, 54, 11, 6, 9],
                ),
            ]
            for td in test_data:
                if not Check.equal(*td):
                    break
            _drevesa = [Drevo(), Drevo()]
            for i in range(1, 12):
                _drevesa.append(Drevo(i, levo=_drevesa[-1], desno=_drevesa[-2]))
                Check.secret(list(_drevesa[-1].po_nivojih()))
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
