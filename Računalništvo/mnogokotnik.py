# =============================================================================
# Mnogokotnik
#
# Mnogokotnik v ravnini lahko predstavimo s seznamom njegovih oglišč
# (pari števil). Na primer seznam
#
#     [(0, 0), (1, 0), (1, 1), (0, 1)]
#
# opisuje kvadrat, seznam
#
#     [(2, 0), (2, 1), (0, 2), (-2, 1), (-2, 0), (0, 1)]
#
# pa opisuje (nekonveksen) šestkotnik.
# =====================================================================@005883=
# 1. podnaloga
# Sestavite razred `Mnogokotnik`, s katerim predstavimo ravninski
# mnogokotnik. Najprej sestavite konstruktor `__init__(self, oglisca)`,
# kjer je `oglisca` seznam njegovih oglišč. Atribut razreda naj bo
# poimenovan `oglisca`.
#
# Nekateri ljudje mnogokotnike zapisujejo tako, da na konec seznama oglišč
# ponovno postavijo prvo oglišče (s tem poudarijo, da je mnogokotnik
# sklenjen). Zgornji kvadrat bi torej zapisali takole:
#
#     [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
#
# V našem razredu `Mnogokotnik` naj se dolžina seznama `oglisca` ujema s
# številom oglišč mnogokotnika. Podvojeno oglišče na koncu seznama naj
# konstruktor po potrebi odstrani. Zgled:
#
#     >>> p = Mnogokotnik([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
#     >>> p.oglisca
#     [(0, 0), (1, 0), (1, 1), (0, 1)]
#
# Predpostavite lahko, da mnogokotniki ne bodo izrojeni (tj. če imata dve
# stranici neprazen presek, sta nujno sosednji, presek pa je natanko
# njuno skupno oglišče).
# =============================================================================
def vektorski_produkt(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


class Mnogokotnik:
    def __init__(self, oglisca):
        self.oglisca = oglisca
        oglisca2 = []
        for i in self.oglisca:
            if not i in oglisca2:
                oglisca2.append(i)
        self.oglisca = oglisca2


# =====================================================================@005884=
# 2. podnaloga
# V razredu `Mnogokotnik` definirajte metodo `obseg(self)`, ki izračuna in
# vrne njegov obseg.
#
#     >>> p = Mnogokotnik([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
#     >>> p.obseg()
#     4.0
# =============================================================================
class Mnogokotnik(Mnogokotnik):
    def obseg(self):
        dolzina = 0
        i = 0
        while i < len(self.oglisca) - 1:
            dolzina += (
                (self.oglisca[i][0] - self.oglisca[i + 1][0]) ** 2
                + (self.oglisca[i][1] - self.oglisca[i + 1][1]) ** 2
            ) ** 0.5
            i += 1
        dolzina += (
            (self.oglisca[0][0] - self.oglisca[len(self.oglisca) - 1][0]) ** 2
            + (self.oglisca[0][1] - self.oglisca[len(self.oglisca) - 1][1]) ** 2
        ) ** 0.5
        return dolzina


# =====================================================================@005885=
# 3. podnaloga
# V razredu `Mnogokotnik` definirajte metodo `ploscina(self)`, ki izračuna
# in vrne njegovo ploščino.
#
# Ploščina mnogokotnika (brez samopresečišč) z oglišči
# $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$ je $|D|/2$, kjer je
# $D = x_1 y_2 - x_2 y_1 + x_2 y_3 - x_3 y_2 + \ldots + x_{n-1} y_n - x_n y_{n-1} + x_n y_1 - x_1 y_n$.
#
#     >>> p = Mnogokotnik([(0, 0), (1, 0), (1, 1), (0, 1)])
#     >>> p.ploscina()
#     1.0
# =============================================================================
class Mnogokotnik(Mnogokotnik):
    def ploscina(self):
        a = 0
        n = len(self.oglisca)
        for i in range(n):
            x, y = self.oglisca[i]
            x2, y2 = self.oglisca[(i + 1) % n]
            a += x * y2 - x2 * y
        return abs(a) / 2


# =====================================================================@005886=
# 4. podnaloga
# V razredu `Mnogokotnik` definirajte metodo `je_konveksen(self)`, ki
# vrne `True`, če je mnogokotnik konveksen in `False` sicer.
#
# Predstavljajte si, da je ravnina na kateri leži mnogokotnik vložena v
# trirazsežni prostor tako, da je $z = 0$. Na bodo $P$, $Q$ in $R$ tri
# zaporedna oglišča mnogokotnika. Naj bo $u$ vektor od $P$ do $Q$ in
# naj bo $v$ vektor od $Q$ do $R$. Potem je $z$-komponenta vektorskega
# produkta vektorjev $u$ in $v$ pozitivna, če $PQR$ "zavije v levo", in
# negativna sicer. Mnogokotnik je konveksen, če vedno "zavijamo v isto
# smer", ko delamo obhod po robu mnogokotnika.
#
#     >>> p = Mnogokotnik([(0, 0), (1, 0), (1, 1), (0, 1)])
#     >>> p.je_konveksen()
#     True
#     >>> q = Mnogokotnik([(2, 0), (2, 1), (0, 2), (-2, 1), (-2, 0), (0, 1)])
#     >>> q.je_konveksen()
#     False
#
# Predpostavite lahko, da nobeni dve zaporedni stranici ne oklepata
# iztegnjenega kota (180 °).
# =============================================================================
def vektorski_produkt(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


class Mnogokotnik(Mnogokotnik):
    def je_konveksen(self):
        n = len(self.oglisca)
        razm = (
            (
                (self.oglisca[1][1] - self.oglisca[0][1])
                / (self.oglisca[1][0] - self.oglisca[0][0])
            )
            - (
                (self.oglisca[0][1] - self.oglisca[len(self.oglisca) - 1][1])
                / (self.oglisca[0][0] - self.oglisca[len(self.oglisca) - 1][0])
            )
        ) > 0
        i = 0
        ogl2 = self.oglisca
        ogl2.extend(self.oglisca[0:3])
        while i < n:
            if not razm == (
                (
                    (self.oglisca[i + 2][1] - self.oglisca[i + 1][1])
                    / (self.oglisca[i + 2][0] - self.oglisca[i + 1][0])
                )
                - (
                    (self.oglisca[i + 1][1] - self.oglisca[i][1])
                    / (self.oglisca[i + 1][0] - self.oglisca[i][0])
                )
            ):
                return False
        return True


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
        ] = "eyJwYXJ0Ijo1ODgzLCJ1c2VyIjo2OTI3fQ:1pMYBj:f9ToCku8RgIh87CO_CTUxTvYpJj1R2L0Ks9POWKTp4E"
        try:
            Check.equal(
                "Mnogokotnik([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]).oglisca",
                [(0, 0), (1, 0), (1, 1), (0, 1)],
            )
            Check.equal(
                "Mnogokotnik([(0, 0), (1, 0), (1, 1), (0, 1)]).oglisca",
                [(0, 0), (1, 0), (1, 1), (0, 1)],
            )
            Check.equal(
                "Mnogokotnik([(2, 0), (2, 1), (0, 2), (-2, 1), (-2, 0), (0,"
                " 1)]).oglisca",
                [(2, 0), (2, 1), (0, 2), (-2, 1), (-2, 0), (0, 1)],
            )
            Check.equal(
                "Mnogokotnik([(2, 0), (2, 1), (0, 2), (-2, 1), (-2, 0), (0, 1), (2,"
                " 0)]).oglisca",
                [(2, 0), (2, 1), (0, 2), (-2, 1), (-2, 0), (0, 1)],
            )
        except:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1ODg0LCJ1c2VyIjo2OTI3fQ:1pMYBj:R1u6RQK08riFQ1VYb71p1z_vX5KpbTxFkPBPg-MAVO4"
        try:
            Check.equal("Mnogokotnik([(0, 0), (1, 0), (1, 1), (0, 1)]).obseg()", 4.0)
            Check.equal(
                "Mnogokotnik([(2, 0), (2, 1), (0, 2), (-2, 1), (-2, 0), (0,"
                " 1)]).obseg()",
                10.94427190999916,
            )
        except:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1ODg1LCJ1c2VyIjo2OTI3fQ:1pMYBj:j4G-WsUGC9lnfLoU11PxEe0L56MMoHKcckUnISMaoTM"
        try:
            Check.equal("Mnogokotnik([(0, 0), (1, 0), (1, 1), (0, 1)]).ploscina()", 1.0)
            Check.equal(
                "Mnogokotnik([(2, 0), (2, 1), (0, 2), (-2, 1), (-2, 0), (0,"
                " 1)]).ploscina()",
                4.0,
            )
            Check.equal(
                "Mnogokotnik([(1, 0), (0.309017, 0.951057), (-0.809017, 0.587785),"
                " (-0.809017, -0.587785), (0.309017, -0.951057)]).ploscina()",
                2.377641895659,
            )
        except:
            Check.error(
                "Testi sprožijo izjemo\n  {0}",
                "\n  ".join(traceback.format_exc().split("\n"))[:-2],
            )

    if Check.part():
        Check.current_part[
            "token"
        ] = "eyJwYXJ0Ijo1ODg2LCJ1c2VyIjo2OTI3fQ:1pMYBj:DBYPlZ1jVvnYHah0I09Y3Gnld3Dg-M8WgMc2TOv0hgI"
        try:
            Check.equal(
                "Mnogokotnik([(0, 0), (1, 0), (1, 1), (0, 1)]).je_konveksen()", True
            )
            Check.equal(
                "Mnogokotnik([(2, 0), (2, 1), (0, 2), (-2, 1), (-2, 0), (0,"
                " 1)]).je_konveksen()",
                False,
            )
            Check.equal(
                "Mnogokotnik([(1, 0), (0.309017, 0.951057), (-0.809017, 0.587785),"
                " (-0.809017, -0.587785), (0.309017, -0.951057)]).je_konveksen()",
                True,
            )
            Check.equal(
                "Mnogokotnik([(4, -4), (10, -6), (14, -4), (16, 2), (12, 8), (8, 10),"
                " (4, 8), (2, 4)]).je_konveksen()",
                True,
            )
            Check.equal(
                "Mnogokotnik([(2, 4), (4, 8), (8, 10), (12, 8), (16, 2), (14, -4), (10,"
                " -6), (4, -4)]).je_konveksen()",
                True,
            )
            Check.equal(
                "Mnogokotnik([(4, -4), (10, -6), (14, -4), (16, 2), (11, 5), (8, 10),"
                " (4, 8), (2, 4)]).je_konveksen()",
                False,
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
