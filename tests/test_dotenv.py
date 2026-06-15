"""The CLI auto-loads ./.env before each command; real environment variables always win."""
import os

from ezpz.config.dotenv import load_dotenv


def test_load_sets_unset_vars_and_skips_noise(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "export ANTHROPIC_API_KEY=sk-ant-123\n"
        'GEMINI_API_KEY="quoted-key"\n'
        "BAD_LINE_NO_EQUALS\n"
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    loaded = load_dotenv(env)
    assert loaded == 2                                       # comment / blank / bad line skipped
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-123"   # `export ` prefix stripped
    assert os.environ["GEMINI_API_KEY"] == "quoted-key"      # surrounding quotes stripped


def test_real_env_wins_unless_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=from-file\n")
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")

    load_dotenv(env)
    assert os.environ["OPENAI_API_KEY"] == "from-shell"      # exported value preserved
    load_dotenv(env, override=True)
    assert os.environ["OPENAI_API_KEY"] == "from-file"       # override forces the .env value


def test_missing_file_is_a_noop(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == 0
