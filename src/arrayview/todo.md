- when on active dim and pressing numbers to enter an index in that dim, i want to see
preview. maybe like dimmed gray and only after enter when confirmed they become white
- inside of the loupe (ctrl or holding mouse) its always the default gray colormap and it
  doesnt seem to chagne when i do change the colormap with the picker from keybind c
- for actual medical images, im not too happy with the ortho view because often one of the
  views is way too small and the way the screen real estate is used is highly inefficient.
  im thinking i need to assume there's always a lot of background with like value 0 (or i
  dunno what it is for CT). and then i should not constrain it to be within square panes and
  allow user to resize any of the three views. this is highly experimental and really
  taylored to medical imaging so im thinking of hacking it together behind some weird
  keybind for now, like shift + 5. what you think? maybe create a visual mockup first before
  changing our code.
