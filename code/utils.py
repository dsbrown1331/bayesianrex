def write_line(_sequence, _writer, newline=True):
    for i,f in enumerate(_sequence):
        if i < len(_sequence) - 1:
            _writer.write(str(f) + ",")
        else:
            if newline:
                _writer.write(str(f) + "\n")
            else:
                _writer.write(str(f))
