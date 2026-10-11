## argparse

- `HelpFormatter`: Formatter for generating usage messages and argument help strings.
- `RawDescriptionHelpFormatter`: Help message formatter which retains any formatting in descriptions.
- `RawTextHelpFormatter`: Help message formatter which retains formatting of all help text.
- `ArgumentDefaultsHelpFormatter`: Help message formatter which adds default values to argument help.
- `MetavarTypeHelpFormatter`: Help message formatter which uses the argument 'type' as the default
- `ArgumentError`: An error from creating or using an argument (optional or positional).
- `ArgumentTypeError`: An error from trying to convert a command line string to a type.
- `Action`: Information about how to convert command line strings to Python objects.
- `BooleanOptionalAction`: 
- `FileType`: Factory for creating file object types
- `Namespace`: Simple object for storing attributes.
- `ArgumentParser`: Object for parsing command line strings into Python objects.

## configparser

- `Error`: Base class for ConfigParser exceptions.
- `NoSectionError`: Raised when no section matches a requested option.
- `DuplicateSectionError`: Raised when a section is repeated in an input source.
- `DuplicateOptionError`: Raised by strict parsers when an option is repeated in an input source.
- `NoOptionError`: A requested option was not found.
- `InterpolationError`: Base class for interpolation-related exceptions.
- `InterpolationMissingOptionError`: A string substitution required a setting which was not available.
- `InterpolationSyntaxError`: Raised when the source text contains invalid syntax.
- `InterpolationDepthError`: Raised when substitutions are nested too deeply.
- `ParsingError`: Raised when a configuration file does not follow legal syntax.
- `MissingSectionHeaderError`: Raised when a key-value pair is found before any section header.
- `MultilineContinuationError`: Raised when a key without value is followed by continuation line
- `Interpolation`: Dummy interpolation that passes the value through with no changes.
- `BasicInterpolation`: Interpolation as implemented in the classic ConfigParser.
- `ExtendedInterpolation`: Advanced variant of interpolation, supports the syntax used by
- `RawConfigParser`: ConfigParser that does not do interpolation.
- `ConfigParser`: ConfigParser implementing interpolation.
- `SectionProxy`: A proxy for a single section from a parser.
- `ConverterMapping`: Enables reuse of get*() methods between the parser and section proxies.

## difflib

- `SequenceMatcher`: SequenceMatcher is a flexible class for comparing pairs of sequences of
- `get_close_matches`: Use SequenceMatcher to return list of the best "good enough" matches.
- `Differ`: Differ is a class for comparing sequences of lines of text, and
- `IS_LINE_JUNK`: Return True for ignorable line: iff `line` is blank or contains a single '#'.
- `IS_CHARACTER_JUNK`: Return True for ignorable character: iff `ch` is a space or tab.
- `unified_diff`: Compare two sequences of lines; generate the delta as a unified diff.
- `context_diff`: Compare two sequences of lines; generate the delta as a context diff.
- `diff_bytes`: Compare `a` and `b`, two sequences of lines represented as bytes rather
- `ndiff`: Compare `a` and `b` (lists of strings); return a `Differ`-style delta.
- `HtmlDiff`: For producing HTML side by side comparison with change highlights.
- `restore`: Generate one of the two sequences that generated a delta.

## fnmatch

- `fnmatch`: Test whether FILENAME matches PATTERN.
- `filter`: Construct a list from those elements of the iterable NAMES that match PAT.
- `fnmatchcase`: Test whether FILENAME matches PATTERN, including case.
- `translate`: Translate a shell PATTERN to a regular expression.

## getopt

- `GetoptError`: 
- `getopt`: getopt(args, options[, long_options]) -> opts, args
- `gnu_getopt`: getopt(args, options[, long_options]) -> opts, args
- `do_longs`: 
- `long_has_args`: 
- `do_shorts`: 
- `short_has_arg`: 

## glob

- `glob`: Return a list of paths matching a `pathname` pattern.
- `iglob`: Return an iterator which yields the paths matching a `pathname` pattern.
- `glob0`: 
- `glob1`: 
- `has_magic`: 
- `escape`: Escape all special characters.
- `translate`: Translate a pathname with shell wildcards to a regular expression.

## optparse

- `OptParseError`: 
- `OptionError`: Raised if an Option instance is created with invalid or
- `OptionConflictError`: Raised if conflicting options are added to an OptionParser.
- `OptionValueError`: Raised if an invalid option value is encountered on the command
- `BadOptionError`: Raised if an invalid option is seen on the command line.
- `AmbiguousOptionError`: Raised if an ambiguous option is seen on the command line.
- `HelpFormatter`: Abstract base class for formatting option help.  OptionParser
- `IndentedHelpFormatter`: Format help with indented section bodies.
- `TitledHelpFormatter`: Format help with underlined section headers.
- `check_builtin`: 
- `check_choice`: 
- `Option`: Instance attributes:
- `Values`: 
- `OptionContainer`: Abstract base class.
- `OptionGroup`: 
- `OptionParser`: Class attributes:

## shlex

- `shlex`: A lexical analyzer class for simple shell-like syntaxes.
- `split`: Split the string *s* using shell-like syntax.
- `join`: Return a shell-escaped string from *split_command*.
- `quote`: Return a shell-escaped version of the string *s*.

## textwrap

- `TextWrapper`: Object for wrapping/filling text.  The public interface consists of
- `wrap`: Wrap a single paragraph of text, returning a list of wrapped lines.
- `fill`: Fill a single paragraph of text, returning a new string.
- `shorten`: Collapse and truncate the given text to fit in the given width.
- `dedent`: Remove any common leading whitespace from every line in `text`.
- `indent`: Adds 'prefix' to the beginning of selected lines in 'text'.
