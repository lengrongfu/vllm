# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

from vllm.utils import FlexibleArgumentParser

# 这是命令函数的基类，所有的命令函数都应该继承这个类；目前主要实现了serve.py 和 openai.py
class CLISubcommand:
    """Base class for CLI argument handlers."""

    name: str

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def validate(self, args: argparse.Namespace) -> None:
        # No validation by default
        pass

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        raise NotImplementedError("Subclasses should implement this method")
