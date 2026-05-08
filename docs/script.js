const showcaseCases = [
  {
    title: "Wukong in Rainy Shibuya",
    image: "assets/cases/wukong-shibuya.jpg",
    status: "verified",
    tags: ["reference", "attribute", "layout"],
    commitments: "7 entities, 9 constraints",
    caption:
      "The Destined One from Black Myth: Wukong standing in the middle of Shibuya Crossing in Tokyo at night, holding the Ruyi Jingu Bang golden staff vertically with one hand. He is wearing signature stone armor with golden accents and a fur-trimmed cape flowing in the wind. The streets are rain-soaked and reflective, neon signs in Japanese kanji illuminate the scene in pink and blue, and pedestrians with transparent umbrellas have stopped to stare. The iconic Shibuya 109 building and Starbucks corner are visible. A giant holographic peach advertisement floats above one of the buildings. Cinematic photography style with shallow depth of field on the character.",
  },
  {
    title: "SpaceX Booster Catch",
    image: "assets/cases/spacex-launch.jpg",
    status: "near-miss",
    tags: ["reference", "relation", "layout"],
    commitments: "4 entities, 8 constraints",
    caption:
      "SpaceX Starship Super Heavy booster being caught mid-air by the Mechazilla chopstick arms on the launch tower at Starbase, Boca Chica, Texas. The booster descends vertically with grid fins deployed, engines firing a landing burn with bright blue-white exhaust. The two massive steel arms extend from the tower and are about to clamp the booster near its grid fins. Early morning golden light, dramatic clouds of steam billowing from the tower base, the Gulf of Mexico visible in the background. Photorealistic wide-angle shot from ground level looking up.",
  },
  {
    title: "Chiikawa Trading Desk",
    image: "assets/cases/chiikawa-trading-floor.jpg",
    status: "verified",
    tags: ["reference", "attribute", "relation"],
    commitments: "7 entities, 10 constraints",
    caption:
      "Chiikawa, Hachiware, and Usagi from Chiikawa sitting at a Wall Street trading desk. Chiikawa is panicking with tears in its eyes while staring at multiple monitors showing red stock charts crashing. Hachiware is nervously biting its paw while holding a phone. Usagi is aggressively smashing the keyboard with a determined expression. They are all wearing tiny ill-fitting suits and ties. The background shows the New York Stock Exchange trading floor with arched windows and American flags, while traders in suits run around in chaos behind them. Anime characters with a realistic financial-floor background and dramatic lighting.",
  },
  {
    title: "Toy Banquet Reveal",
    image: "assets/cases/jellycat-banquet.jpg",
    status: "verified",
    tags: ["attribute", "relation", "layout"],
    commitments: "9 entities, 10 constraints",
    caption:
      "A Jellycat eggplant plush toy seated at a formal table in a Michelin three-star restaurant. A waiter in white gloves lifts a silver cloche to reveal a tiny Jellycat mushroom plush on the plate. The eggplant has a cloth napkin tucked into its front. Across the table, a Jellycat lobster plush looks horrified at a menu that reads \"Lobster Thermidor\" in cursive. Crystal chandeliers hang overhead, and other diners are different Jellycat plushies. Warm candlelight, photorealistic product photography style, soft bokeh background.",
  },
  {
    title: "Crazy Dave Lecture Hall",
    image: "assets/cases/crazy-dave-classroom.jpg",
    status: "verified",
    tags: ["reference", "attribute", "layout"],
    commitments: "8 entities, 12 constraints",
    caption:
      "Crazy Dave from Plants vs Zombies standing at a university lecture podium, wearing his signature saucepan on his head and wild beard, enthusiastically gesturing at a chalkboard filled with complex equations labeled \"Wabi Babu Linguistic Theory\". He is wearing a tweed blazer over his usual outfit. The audience consists of Peashooters sitting in lecture hall seats and taking notes with tiny pencils, a Sunflower in the front row raising its leaf to ask a question, and a Wall-nut sleeping in the back row. A Zombie wearing glasses and a graduate cap is the teaching assistant standing by the door. Pixar-style 3D rendering with warm indoor lighting.",
  },
  {
    title: "Culinary Tasting Table",
    image: "assets/cases/culinary-tasting-table-natural.png",
    status: "preview",
    tags: ["attribute", "relation", "layout"],
    commitments: "10 entities, 6 constraints",
    caption:
      "Create a beautiful square premium overhead editorial food photograph of a luxury East Asian tasting table. A matte black stone platter sits exactly in the center on a warm wood counter. On the platter, arrange exactly seven jewel-like tasting dishes in a loose crescent: glossy seared fish, a folded dumpling, a bright green vegetable roll, a red radish fan, a tiny golden custard, a black sesame bite, and a translucent citrus jelly. A glass cloche is being lifted above the platter, releasing delicate aromatic smoke without hiding the dishes. Around the platter, include a lacquer soup bowl, ceramic tea cup, black chopsticks, folded linen, polished wood grain, soft shadows, and fine-dining warmth. Avoid readable text, logos, watermarks, messy clutter, distorted plates, and the wrong number of tasting dishes.",
  },
  {
    title: "Prism Product Campaign",
    image: "assets/cases/prism-product-natural.png",
    status: "preview",
    tags: ["attribute", "layout"],
    commitments: "5 entities, 5 constraints",
    caption:
      "Create a beautiful square premium commercial product advertisement for a fictional transparent prism camera device. Center the product in the frame: a crystal-clear lens device with polished metal edges, a luminous inner prism, and no readable brand text. Behind and to the left, place a matte black product box with no readable text. Exactly three colored light beams pass through the transparent lens: cyan, magenta, and amber, creating crisp refractions and clean reflections on a mirror-polished studio surface. Avoid readable text, real brand logos, watermarks, clutter, and more or fewer than three light beams.",
  },
  {
    title: "Glass-Roof Library",
    image: "assets/cases/glass-roof-library-natural.png",
    status: "preview",
    tags: ["attribute", "layout"],
    commitments: "7 entities, 5 constraints",
    caption:
      "Create a beautiful square architectural interior visualization of a sunlit modern atrium library. A curved clear glass ceiling spans the top, casting soft morning light into the space. Along the back wall, show exactly three floating reading platforms stacked vertically, each with slim brass railings and warm walnut shelving. At the center of the travertine floor, place an indoor olive tree in a circular stone planter. Use refined materials: travertine floor, walnut shelves, brass railings, clear glass, soft shadows, calm luxury-hotel proportions, and airy depth. Keep the scene natural and free of readable text or signage.",
  },
  {
    title: "Astronomical Observatory",
    image: "assets/cases/astronomical-observatory-natural.png",
    status: "preview",
    tags: ["attribute", "relation"],
    commitments: "7 entities, 6 constraints",
    caption:
      "Create a beautiful square high-end futuristic astronomical observatory at night. A large telescope points through an open dome toward a deep starry sky and faint Milky Way. In the center-right, place a transparent holographic star map with orbit lines but no readable text or labels. On the star map, show exactly four glowing exoplanet markers in distinct colors orbiting around a small central star diagram. A sleek robotic lab assistant stands beside the display and points at the markers without covering them. Avoid readable text, logos, watermarks, clutter blocking the star map, wrong number of planet markers, and toy-like equipment.",
  },
  {
    title: "Tang Lantern Festival",
    image: "assets/cases/tang-lantern-festival-natural.png",
    status: "preview",
    tags: ["attribute", "relation", "layout"],
    commitments: "7 entities, 6 constraints",
    caption:
      "Create a stunning square cinematic historical-culture scene inspired by a Tang dynasty lantern festival at night. In the foreground, show exactly three performers wearing elegant Tang-inspired robes, silk sashes, and ornate hair ornaments. Behind them, place a warm wooden palace pavilion with curved eaves, carved railings, layered roof details, and glowing lantern light. A red lantern arch spans above dark water in the upper background, with reflections on wet stone pavement. Add a small moonlit stone bridge to the left of the performers, silk banners with decorative abstract pattern marks, drifting mist, gold-red light, and rich fabric detail. Avoid readable text, modern objects, logos, watermarks, inaccurate random costumes, and more or fewer than three foreground performers.",
  },
  {
    title: "Space-Heist Museum",
    image: "assets/cases/space-museum-heist.jpg",
    status: "verified",
    tags: ["relation", "layout"],
    commitments: "8 entities, 6 constraints",
    caption:
      "Create a whimsical sci-fi museum scene about a failed space-heist rehearsal. At the center, a young curator in a silver jacket holds a replica of the Voyager Golden Record above a glass display case. To the curator's left, a small robot thief points a flashlight at a miniature James Webb Space Telescope inside the case. To the curator's right, a child detective in a yellow raincoat holds an open star map. Behind all three characters, a large wall mural shows the Pillars of Creation nebula. The scene should be playful, cinematic, and highly detailed.",
  },
  {
    title: "Toy Shelf Spatial Layout",
    image: "assets/cases/toy-shelf-layout.jpg",
    status: "preview",
    tags: ["layout"],
    commitments: "6 entities, 1 grouped layout constraint",
    caption:
      "Generate a polished toy-store display shelf scene in a semi-realistic illustration style. The shelf has three levels. Put a red race car on the bottom shelf, a blue robot toy on the top shelf, and a yellow teddy bear on the middle shelf. Place a small green dinosaur toy to the left of the teddy bear and a purple gift box to the right of the teddy bear. The shelf should be front-facing, neatly organized, and all toys should be easy to identify.",
  },
  {
    title: "Award Table Spatial Layout",
    image: "assets/cases/award-table-layout.jpg",
    status: "preview",
    tags: ["layout", "attribute"],
    commitments: "6 entities, 1 grouped layout constraint",
    caption:
      "Generate a realistic award-ceremony preparation table before guests arrive. A rectangular table is viewed from the front. Put a tall silver trophy on the far left side of the table, a folded red ribbon in front of the trophy, and a stack of three certificates on the right side. Place a blue name card behind the certificates and a small bouquet on top of the table between the trophy and certificates. The background has soft stage lights and a formal ceremony atmosphere.",
  },
  {
    title: "Nezha Hotpot Birthday Table",
    image: "assets/cases/nezha-hotpot-birthday-table.png",
    status: "verified",
    tags: ["reference", "attribute", "relation"],
    commitments: "6 entities, 5 constraints",
    caption:
      "Create a cinematic square image of Nezha and Ao Bing from Ne Zha 2 celebrating a birthday at a lively Chinese hotpot restaurant. Nezha sits in the left foreground, smirking while holding a glowing Qiankun Ring above the table. Ao Bing sits across from him on the right, calm and elegant, with pale blue dragon motifs in his outfit. A red Huntian Ling ribbon wraps around the back of Nezha's chair. In the exact center of the table is a split spicy-and-clear yin-yang hotpot, with a small birthday cake behind it. A warm sign on the back wall reads only 'Happy Birthday'. Keep all objects visible and avoid extra main characters.",
  },
  {
    title: "Langlang Mountain Weasel Courier",
    image: "assets/cases/langlang-weasel-delivery.png",
    status: "verified",
    tags: ["reference", "attribute", "layout"],
    commitments: "5 entities, 4 constraints",
    caption:
      "Create a warm Chinese animated-road-movie style square image inspired by Nobody / The Little Monsters of Langlang Mountain. A weasel-like courier stands at a small mountain delivery stop wearing a simple gray travel robe and carrying a parcel basket. Beside the courier are several wrapped packages and a handwritten delivery slip. A wooden sign in the background reads only 'Langlang Mountain Station'. Bamboo, misty hills, and a narrow mountain path frame the scene. Keep the mood humorous, tired, and hopeful.",
  },
  {
    title: "Snow King Tea Shop Parade",
    image: "assets/cases/mixue-snow-king-parade.png",
    status: "verified",
    tags: ["reference", "attribute", "layout"],
    commitments: "5 entities, 4 constraints",
    caption:
      "Create a bright square illustration of the Mixue Snow King mascot leading a cheerful tea-shop parade in front of a Mixue Ice Cream & Tea storefront. The Snow King stands in the center holding an ice cream cone and pulling a small drink cart. Around the cart are four excited fans holding signs and balloons. The red storefront sign is visible in the background, with a festive crowd and clean street layout. Keep the mascot, cart, shop sign, and crowd clearly separated.",
  },
  {
    title: "Elden Ring Grace Rest Stop",
    image: "assets/cases/elden-ring-grace-rest-stop.png",
    status: "verified",
    tags: ["reference", "relation", "layout"],
    commitments: "5 entities, 4 constraints",
    caption:
      "Create a moody cinematic square image of an Elden Ring Tarnished resting at a modern highway service area at night. The armored Tarnished sits on a metal bench in the left foreground, helmet resting beside them. A glowing golden Site of Grace spiral rises from the pavement directly in front of the bench. Torrent, the spectral horned steed, is tied near an electric scooter charging station on the right. On a small table between the Tarnished and the Site of Grace are a torn map fragment, a flask, and a convenience-store cup. A fluorescent minimart glows in the background. Keep the scene realistic but with clear fantasy cues.",
  },
  {
    title: "Baldur's Gate 3 Hotpot Party",
    image: "assets/cases/baldurs-gate-hotpot-party.png",
    status: "verified",
    tags: ["reference", "attribute", "relation"],
    commitments: "7 entities, 4 constraints",
    caption:
      "Create a detailed square fantasy tavern image of Astarion, Shadowheart, and Karlach from Baldur's Gate 3 sharing a Chinese hotpot party. Astarion sits on the left, pale and elegant, leaning away from a garlic sauce bowl with a suspicious expression. Shadowheart sits at the center holding a small black d20 die above a parchment map. Karlach sits on the right beside a running metal fan, smiling with red tiefling skin, horns, and glowing engine-like chest details. In the center is a bubbling hotpot; on the table are chopsticks, a spellbook, and three labeled dipping bowls. Keep all three characters visible and distinct.",
  },
  {
    title: "Chainsaw Man Convenience Shift",
    image: "assets/cases/chainsaw-man-convenience-shift.png",
    status: "verified",
    tags: ["reference", "attribute", "layout"],
    commitments: "6 entities, 5 constraints",
    caption:
      "Create a square anime-style night-shift convenience store scene featuring Denji, Pochita, Power, and Aki from Chainsaw Man. Denji stands behind the checkout counter wearing a convenience-store uniform and name tag, holding a barcode scanner. Pochita sits on the counter beside the register. Power stands in the left aisle hugging a messy pile of snacks. Aki is in the rear right, neatly arranging canned coffee on a shelf. The foreground has a wet floor sign and a mop bucket; the background has a bento fridge and a no-smoking sign. Keep the store layout clear and avoid extra main characters.",
  },
  {
    title: "Spy Family Parent-Teacher Meeting",
    image: "assets/cases/spy-family-parent-teacher-meeting.png",
    status: "verified",
    tags: ["reference", "relation", "layout"],
    commitments: "6 entities, 4 constraints",
    caption:
      "Create a polished anime-style square image of Loid, Yor, and Anya from Spy x Family at a parent-teacher meeting in an elegant school classroom. Anya sits in a small chair in the center foreground, looking sideways with a mischievous telepathic expression. Loid sits on the left holding a neat folder of school documents. Yor sits on the right with perfect posture, but a steel pen is accidentally bent in her hand. A blackboard behind them shows a simple meeting schedule with three bullet lines, and a teacher's desk is visible in the background. Keep the family triangle composition clear.",
  },
  {
    title: "Jujutsu Kaisen Subway Incident Board",
    image: "assets/cases/jujutsu-kaisen-subway-incident-board.png",
    status: "verified",
    tags: ["reference", "relation", "layout"],
    commitments: "6 entities, 5 constraints",
    caption:
      "Create a dramatic square anime-style image of Yuji Itadori, Megumi Fushiguro, and Nobara Kugisaki from Jujutsu Kaisen standing in a subway station command corner. A portable whiteboard with a subway line map stands behind them. Yuji is in the center pointing at a red curse marker on the map. Megumi stands on the left holding a folded ticket route sheet. Nobara stands on the right with a hammer resting on her shoulder. Gojo's black blindfold is pinned to the upper corner of the board. A convenience-store bag and three talisman stickers sit on the table below the board. Keep the subway platform lights and tiled wall visible.",
  },
  {
    title: "Haikyu Exam Week Practice",
    image: "assets/cases/haikyu-exam-week-practice.png",
    status: "verified",
    tags: ["reference", "relation", "layout"],
    commitments: "6 entities, 5 constraints",
    caption:
      "Create a bright square anime-style gym scene of Hinata Shoyo, Kageyama Tobio, and Tsukishima Kei from Haikyu during exam week volleyball practice. Hinata is in the left foreground jumping near the net while holding a vocabulary card. Kageyama stands in the center tossing a volleyball upward with one hand and holding a math worksheet in the other. Tsukishima sits on a bench on the right reading a textbook with headphones around his neck. On the floor near the net are three flashcards and one volleyball. A scoreboard in the background reads only 'Mock Test 80'. Keep the volleyball court lines and net clear.",
  },
];

const scopebenchCases = [
  {
    title: "Cartoon 004",
    image: "assets/scopebench_cases/1-cartoon_004.jpg",
    status: "verified",
    tags: ["cartoon"],
    commitments: "6 entities, 7 constraints",
    caption:
      'Create an image of Gretel Grant-Gomez from Hamster & Gretel standing on a small stage. She is positioned in the midground, firmly holding a tall microphone stand. Behind her, a large display board rests against a heavy stage curtain in the background. A black floor monitor sits in the lower foreground near the stage edge. The display board features the short text "LIVE" in bold letters.',
    score: 1.0,
  },
  {
    title: "Cartoon 012",
    image: "assets/scopebench_cases/1-cartoon_012.jpg",
    status: "verified",
    tags: ["cartoon"],
    commitments: "8 entities, 8 constraints",
    caption:
      'Create an image of Doug from Doug Unplugs standing in a robot testing room. He is positioned at a control panel in the midground, holding a cable connector toward a small robot on the table. A status screen behind the robot displays the word "GO". A cable spool sits in the lower foreground, and a storage cabinet is visible in the background.',
    score: 1.0,
  },
  {
    title: "Cartoon 017",
    image: "assets/scopebench_cases/1-cartoon_017.jpg",
    status: "verified",
    tags: ["cartoon"],
    commitments: "8 entities, 6 constraints",
    caption:
      "Create an image of Garbage from Dogs in Space standing behind a large metal workspace table. He is holding a rolled-up blueprint in his paws. An open toolbox rests on the table next to a scattered pile of metal gears. In the background, a glowing desk lamp is placed near a tall storage shelf. A tilted stool sits in the lower foreground.",
    score: 1.0,
  },
  {
    title: "Cartoon 028",
    image: "assets/scopebench_cases/1-cartoon_028.jpg",
    status: "verified",
    tags: ["cartoon"],
    commitments: "7 entities, 9 constraints",
    caption:
      'Create an image of Dorg Van Dango from Dorg Van Dango presenting at a school science fair. He is positioned in the midground, holding a microphone beside a table with a tabletop model on it. A display board stands behind the model with the word "INFO" displayed on it. Three labeled jars are lined up along the lower foreground edge of the table. A banner hangs in the background above the table.',
    score: 1.0,
  },
  {
    title: "Cartoon 043",
    image: "assets/scopebench_cases/1-cartoon_043.jpg",
    status: "verified",
    tags: ["cartoon"],
    commitments: "8 entities, 6 constraints",
    caption:
      "Create an image of Lance from Sym-Bionic Titan standing in a hangar repair bay. He is positioned in the midground, leaning against a work platform. A folded tarp lies across a metal crate in the lower foreground. Three tool cases are stacked beside the platform. A warning light is mounted in the background above a service door.",
    score: 1.0,
  },
  {
    title: "Cartoon 046",
    image: "assets/scopebench_cases/1-cartoon_046.jpg",
    status: "verified",
    tags: ["cartoon"],
    commitments: "9 entities, 7 constraints",
    caption:
      "Create an image of Vendetta from Making Fiends standing at a potion workbench. She is holding a pencil over an open recipe card. A cauldron sits on the workbench beside three potion bottles. A shelf of jars is visible in the background. A hanging lamp is suspended above the cauldron, and a stool sits in the lower foreground.",
    score: 1.0,
  },
  {
    title: "Cartoon 049",
    image: "assets/scopebench_cases/1-cartoon_049.jpg",
    status: "verified",
    tags: ["cartoon"],
    commitments: "7 entities, 5 constraints",
    caption:
      'Create an image of Fangbone from Fangbone! standing behind a wooden booth counter on a raised platform. He holds a large wooden club raised in his right hand. A display board rests on the counter, featuring the short text "QUEST" in bold letters. The lower foreground shows dark audience silhouettes watching the platform. A heavy stage curtain hangs in the background, framing the scene.',
    score: 1.0,
  },
  {
    title: "Game 001",
    image: "assets/scopebench_cases/2-game_001.jpg",
    status: "verified",
    tags: ["game"],
    commitments: "7 entities, 6 constraints",
    caption:
      "Create an image of Curly Brace from Cave Story resting in a mechanical workshop. She is seated on a large metal crate in the foreground, holding a heavy rifle across her lap. Directly behind her in the midground is a sturdy workbench with a glowing monitor resting on top of it. A tall tool rack stands against the wall in the background.",
    score: 1.0,
  },
  {
    title: "Game 002",
    image: "assets/scopebench_cases/2-game_002.jpg",
    status: "verified",
    tags: ["game"],
    commitments: "6 entities, 7 constraints",
    caption:
      "Create an image of Elster from SIGNALIS and Ariane Yeong from SIGNALIS inside a spacecraft cabin. Elster is standing in the foreground, facing a wide control console. Ariane Yeong is seated on a chair in the midground, looking toward Elster. A large screen is mounted on the wall directly behind the console, displaying a glowing red grid. A closed, heavy metal door is visible in the background on the right side of the room.",
    score: 1.0,
  },
  {
    title: "Game 003",
    image: "assets/scopebench_cases/2-game_003.jpg",
    status: "verified",
    tags: ["game"],
    commitments: "5 entities, 6 constraints",
    caption:
      "Create an image of Jill Stingray from VA-11 Hall-A and Dana Zane from VA-11 Hall-A in a bar setting. Jill Stingray is standing in the midground behind a sleek bar counter, holding a cocktail shaker in both hands. Dana Zane is seated on a tall stool in the foreground across the bar counter, leaning forward and looking directly at Jill. A bright neon sign hangs on the wall in the background, illuminating the space.",
    score: 1.0,
  },
  {
    title: "Game 021",
    image: "assets/scopebench_cases/2-game_021.jpg",
    status: "verified",
    tags: ["game"],
    commitments: "7 entities, 7 constraints",
    caption:
      "Create an image of Mashiro Mito from Tayutama standing in the foreground of a traditional courtyard. She is holding a sweeping broom with both hands and looking directly at the viewer. A large red torii gate stands prominently in the midground behind her. To her left, a stone lantern rests on the ground surrounded by scattered fallen leaves. The wooden structure of a shrine building is visible in the background.",
    score: 1.0,
  },
  {
    title: "Game 022",
    image: "assets/scopebench_cases/2-game_022.jpg",
    status: "verified",
    tags: ["game"],
    commitments: "6 entities, 6 constraints",
    caption:
      "Create an image of Mare S. Ephemeral from Hoshizora no Memoria standing on an observation deck in the foreground. She is looking through a large telescope that points upward toward a starry sky in the background. A glowing lantern rests on the deck beside her. She is holding an open notebook.",
    score: 1.0,
  },
  {
    title: "Game 026",
    image: "assets/scopebench_cases/2-game_026.jpg",
    status: "verified",
    tags: ["game"],
    commitments: "6 entities, 6 constraints",
    caption:
      "Create an image of Setsuna Ogiso from White Album 2 performing on a stage. She is standing in the foreground, holding a microphone stand with both hands. A stage monitor rests on the floor directly in front of her. In the background, a heavy curtain is illuminated by a bright spotlight. A closed guitar case is visible to the left of the stage monitor.",
    score: 1.0,
  },
  {
    title: "Game 048",
    image: "assets/scopebench_cases/2-game_048.jpg",
    status: "verified",
    tags: ["game"],
    commitments: "5 entities, 6 constraints",
    caption:
      "Create an image of Hikari from Arcaea standing on a transparent glass platform in the foreground. She is reaching her hand toward a large floating crystal positioned in the midground. Behind her, a glowing archway frames the scene. A white staircase descends into the lower background.",
    score: 1.0,
  },
  {
    title: "Game 050",
    image: "assets/scopebench_cases/2-game_050.jpg",
    status: "verified",
    tags: ["game"],
    commitments: "7 entities, 6 constraints",
    caption:
      "Create an image of Groal the Great from Hollow Knight: Silksong working in a dimly lit blacksmith workshop. Groal the Great is standing in the foreground, holding a large hammer. Directly in front of him rests a heavy iron anvil. In the midground, a bright forge fire is burning inside a stone pillar. A large metal shield is leaning against the stone pillar in the background.",
    score: 1.0,
  },
  {
    title: "Sports 003",
    image: "assets/scopebench_cases/3-sports_003.jpg",
    status: "verified",
    tags: ["sports"],
    commitments: "8 entities, 9 constraints",
    caption:
      "Create an image of Sorato Anraku clipping a safety rope into a quickdraw on an overhanging indoor climbing wall. A belayer silhouette stands on a crash mat below while holding the rope. A route tag is fixed near the base of the wall, and a large volume hold protrudes in the lower foreground. The climbing wall fills the background behind the rope path.",
    score: 1.0,
  },
  {
    title: "Sports 008",
    image: "assets/scopebench_cases/3-sports_008.jpg",
    status: "verified",
    tags: ["sports"],
    commitments: "6 entities, 7 constraints",
    caption:
      "Create an image of Hania El Hammamy reaching low across a squash court with her squash racket extended. A squash ball sits just above the court floor near the racket. A side wall runs along one edge of the scene, while the front wall stands in the background. The racket and ball occupy the foreground, and the front wall creates the rear layer.",
    score: 1.0,
  },
  {
    title: "Sports 013",
    image: "assets/scopebench_cases/3-sports_013.jpg",
    status: "verified",
    tags: ["sports"],
    commitments: "6 entities, 6 constraints",
    caption:
      "Create an image of Anders Skaarup Rasmussen suspended above the badminton court during a jump smash. He holds a badminton racket high and reaches toward a shuttlecock above the racket. A taut net stretches across the midground. A generic opponent stands on the far side of the net. A court boundary line is visible below the airborne player.",
    score: 1.0,
  },
  {
    title: "Sports 018",
    image: "assets/scopebench_cases/3-sports_018.jpg",
    status: "verified",
    tags: ["sports"],
    commitments: "6 entities, 6 constraints",
    caption:
      "Create an image of Shin Yubin leaning over the near edge of a table tennis table. She grips a table tennis paddle during a backhand stroke, and a table tennis ball hovers above the paddle surface. A taut net divides the table across the midground. A low side barrier encloses the background. The table tennis table occupies the foreground.",
    score: 1.0,
  },
  {
    title: "Sports 023",
    image: "assets/scopebench_cases/3-sports_023.jpg",
    status: "verified",
    tags: ["sports"],
    commitments: "6 entities, 6 constraints",
    caption:
      "Create an image of Tommaso Marini paused between fencing actions on a long fencing piste. He holds a fencing mask under one arm while gripping a fencing weapon with the blade tip pointed down toward the fencing piste. A masked opponent waits at the opposite end in the background. The fencing piste stretches across the foreground. A scoring box rests near the edge of the fencing piste.",
    score: 1.0,
  },
  {
    title: "Sports 031",
    image: "assets/scopebench_cases/3-sports_031.jpg",
    status: "verified",
    tags: ["sports"],
    commitments: "6 entities, 8 constraints",
    caption:
      "Create an image of Sarah Hildebrandt working a hand-fighting drill on a wrestling mat. She is in a low stance in the foreground, reaching toward a generic training partner. The center circle is painted beneath the two wrestlers. An ankle band wraps around her lower leg. A mat boundary line stretches across the background.",
    score: 1.0,
  },
  {
    title: "Sports 036",
    image: "assets/scopebench_cases/3-sports_036.jpg",
    status: "verified",
    tags: ["sports"],
    commitments: "5 entities, 8 constraints",
    caption:
      "Create an image of Ricarda Funk in a slalom kayak on a turbulent water channel. She leans to one side while gripping a paddle, with the paddle touching the water. Water spray splashes around the kayak. Two gate poles hang above the water channel in the midground. The athlete and kayak occupy the foreground.",
    score: 1.0,
  },
  {
    title: "Sports 042",
    image: "assets/scopebench_cases/3-sports_042.jpg",
    status: "verified",
    tags: ["sports"],
    commitments: "9 entities, 8 constraints",
    caption:
      "Create an image of Karolien Florijn training on an indoor rowing ergometer. She sits on the sliding seat while gripping the handle with both hands. A digital monitor is mounted in front of the ergometer. A water bottle stands beside the ergometer rail in the lower foreground, and a wall mirror reflects the ergometer frame in the background.",
    score: 1.0,
  },
  {
    title: "Entertainment 003",
    image: "assets/scopebench_cases/4-entertainment_003.jpg",
    status: "verified",
    tags: ["entertainment"],
    commitments: "4 entities, 4 constraints",
    caption:
      "Create an image of Xyla Foxlin standing in the foreground at a workbench. She is assembling a small mechanical device with both hands. On the workbench there are scattered tools, small parts, a soldering iron, and a partially completed project beside the tools. Behind her are shelves filled with equipment and storage bins.",
    score: 1.0,
  },
  {
    title: "Entertainment 038",
    image: "assets/scopebench_cases/4-entertainment_038.jpg",
    status: "verified",
    tags: ["entertainment"],
    commitments: "5 entities, 4 constraints",
    caption:
      "Create an image of Mau P standing behind a festival DJ booth in the foreground. He has one hand raised and the other hand above the controls. Overhead stage lighting rigs and an LED screen are positioned behind him. A crowd area is visible in the lower background.",
    score: 1.0,
  },
  {
    title: "Competition 006",
    image: "assets/scopebench_cases/5-competition_006.jpg",
    status: "verified",
    tags: ["competition"],
    commitments: "5 entities, 5 constraints",
    caption:
      "Create an image after the 2024 Super Bowl from the end-zone side. Show a large readable scoreboard beyond the field edge, with a goalpost standing between the field and the board. A tunnel rail crosses the lower foreground, and a seating block rises behind the scoreboard.",
    score: 1.0,
  },
  {
    title: "Competition 010",
    image: "assets/scopebench_cases/5-competition_010.jpg",
    status: "verified",
    tags: ["competition"],
    commitments: "5 entities, 5 constraints",
    caption:
      "Create an image of a post-game press conference table after the 2024 NCAA Men's Basketball Championship Game. A press conference result placard on the press table shows the final score. Two tabletop microphones sit beside the placard, a championship backdrop fills the background, and a strip of hardwood floor spans the lower foreground.",
    score: 1.0,
  },
  {
    title: "Competition 015",
    image: "assets/scopebench_cases/5-competition_015.jpg",
    status: "verified",
    tags: ["competition"],
    commitments: "7 entities, 6 constraints",
    caption:
      "Create an image after the 2024 Wimbledon Men's Final showing a large readable set scoreboard built into a grass-court wall near the umpire chair. The tennis net cuts diagonally across the midground, and the grass court edge runs through the lower foreground. A towel box sits below the scoreboard, with spectator seating visible above the wall.",
    score: 1.0,
  },
  {
    title: "Ceremony 001",
    image: "assets/scopebench_cases/6-ceremony_001.jpg",
    status: "verified",
    tags: ["ceremony"],
    commitments: "6 entities, 7 constraints",
    caption:
      "Create an image of the Directing award presentation at the 2024 Oscars. On the left side of the background, a large winner screen displays the ceremony category and winner name line. The recipient stands beside a stage table in the right midground, holding the award trophy. An envelope rests on the table, and the stage edge runs across the lower foreground.",
    score: 1.0,
  },
  {
    title: "Ceremony 017",
    image: "assets/scopebench_cases/6-ceremony_017.jpg",
    status: "verified",
    tags: ["ceremony"],
    commitments: "9 entities, 8 constraints",
    caption:
      "Create an image of the Cinematography recipient at the 2024 BAFTA Film Awards beside a camera display setup. The recipient stands next to a camera rig while a camera slate sits on an equipment case. A winner screen mounted above the rig shows the category and winner name line. A light stand is visible in the foreground, and a tape mark lies on the stage floor near the case.",
    score: 1.0,
  },
  {
    title: "Ceremony 045",
    image: "assets/scopebench_cases/6-ceremony_045.jpg",
    status: "verified",
    tags: ["ceremony"],
    commitments: "6 entities, 7 constraints",
    caption:
      "Create an image of the Best African Music Performance recipient at the 2024 GRAMMY Awards in a backstage press corner. A result board on one side behind the press table displays the category and winner name line. The recipient stands behind the table with the press credential near the table center. A row of microphones points across the table toward the recipient, and a camera bag sits under one side of the table.",
    score: 1.0,
  },
  {
    title: "Ceremony 049",
    image: "assets/scopebench_cases/6-ceremony_049.jpg",
    status: "verified",
    tags: ["ceremony"],
    commitments: "6 entities, 7 constraints",
    caption:
      "Create an image of the Outstanding Performance by a Female Actor in a Supporting Role recipient at the 2024 Screen Actors Guild Awards beside a large winner screen. The winner screen on one side of the background displays the category and winner name line. The recipient stands near a stage table on the other side and holds the result card. A floor monitor sits in front of the table, and the stage edge runs along the bottom of the image.",
    score: 1.0,
  },
];

const genArenaCaseMetadata = {
  "Cartoon 004": { title: "Gretel Stage Broadcast", tags: ["reference", "attribute", "layout"] },
  "Cartoon 012": { title: "Doug Robot Cable Test", tags: ["reference", "relation", "layout"] },
  "Cartoon 017": { title: "Garbage Workshop Blueprint", tags: ["reference", "attribute", "relation"] },
  "Cartoon 028": { title: "Dorg Science Fair Booth", tags: ["reference", "attribute", "layout"] },
  "Cartoon 043": { title: "Lance Hangar Repair Bay", tags: ["reference", "attribute", "layout"] },
  "Cartoon 046": { title: "Vendetta Potion Workbench", tags: ["reference", "attribute", "relation"] },
  "Cartoon 049": { title: "Fangbone Quest Booth", tags: ["reference", "attribute", "layout"] },
  "Game 001": { title: "Curly Brace Workshop Rest", tags: ["reference", "attribute", "layout"] },
  "Game 002": { title: "SIGNALIS Cabin Console", tags: ["reference", "relation", "layout"] },
  "Game 003": { title: "VA-11 Hall-A Bar Exchange", tags: ["reference", "relation", "layout"] },
  "Game 021": { title: "Mashiro Shrine Courtyard", tags: ["reference", "attribute", "layout"] },
  "Game 022": { title: "Mare Observatory Notebook", tags: ["reference", "relation", "layout"] },
  "Game 026": { title: "Setsuna Stage Performance", tags: ["reference", "attribute", "layout"] },
  "Game 048": { title: "Hikari Crystal Platform", tags: ["reference", "relation", "layout"] },
  "Game 050": { title: "Silksong Blacksmith Forge", tags: ["reference", "attribute", "relation"] },
  "Sports 003": { title: "Anraku Climbing Quickdraw", tags: ["reference", "relation", "layout"] },
  "Sports 008": { title: "Hammamy Squash Reach", tags: ["reference", "relation", "layout"] },
  "Sports 013": { title: "Rasmussen Jump Smash", tags: ["reference", "relation", "layout"] },
  "Sports 018": { title: "Shin Yubin Backhand", tags: ["reference", "relation", "layout"] },
  "Sports 023": { title: "Marini Fencing Piste", tags: ["reference", "attribute", "layout"] },
  "Sports 031": { title: "Hildebrandt Wrestling Drill", tags: ["reference", "relation", "layout"] },
  "Sports 036": { title: "Ricarda Funk Slalom Gate", tags: ["reference", "relation", "layout"] },
  "Sports 042": { title: "Florijn Rowing Ergometer", tags: ["reference", "attribute", "layout"] },
  "Entertainment 003": { title: "Xyla Workshop Build", tags: ["reference", "attribute", "relation"] },
  "Entertainment 038": { title: "Mau P Festival DJ Booth", tags: ["reference", "relation", "layout"] },
  "Competition 006": { title: "Super Bowl End-Zone Scoreboard", tags: ["reference", "attribute", "layout"] },
  "Competition 010": { title: "NCAA Basketball Press Table", tags: ["reference", "attribute", "relation"] },
  "Competition 015": { title: "Wimbledon Men's Set Board", tags: ["reference", "attribute", "layout"] },
  "Ceremony 001": { title: "Oscars Directing Winner Screen", tags: ["reference", "attribute", "relation"] },
  "Ceremony 017": { title: "BAFTA Cinematography Camera Display", tags: ["reference", "attribute", "relation"] },
  "Ceremony 045": { title: "Grammys Press Corner", tags: ["reference", "attribute", "relation"] },
  "Ceremony 049": { title: "SAG Awards Winner Screen", tags: ["reference", "attribute", "layout"] },
};

const genArenaCases = scopebenchCases.map((item) => {
  const category = item.tags[0];
  return { ...item, category, ...genArenaCaseMetadata[item.title] };
});

const cases = [...showcaseCases, ...genArenaCases];

const filterOrder = [
  "all",
  "reference",
  "attribute",
  "relation",
  "layout",
  "cartoon",
  "game",
  "sports",
  "entertainment",
  "competition",
  "ceremony",
];
let activeFilter = "all";

function labelForFilter(filter) {
  return filter
    .split("-")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function visibleCases() {
  if (activeFilter === "all") return cases;
  return cases.filter((item) => caseFilters(item).includes(activeFilter));
}

function caseFilters(item) {
  return [...item.tags, item.category].filter(Boolean);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderFilters() {
  const filters = document.querySelector("#filters");
  if (!filters) return;

  filters.innerHTML = filterOrder
    .map((filter) => {
      const count =
        filter === "all"
          ? cases.length
          : cases.filter((item) => caseFilters(item).includes(filter)).length;
      return `<button class="filter-button ${
        filter === activeFilter ? "is-active" : ""
      }" type="button" data-filter="${filter}">${labelForFilter(filter)} ${count}</button>`;
    })
    .join("");

  filters.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      activeFilter = button.dataset.filter;
      renderFilters();
      renderGallery();
    });
  });
}

function renderGallery() {
  const grid = document.querySelector("#galleryGrid");
  if (!grid) return;

  grid.innerHTML = visibleCases()
    .map(
      (item) => `
      <article class="case-card">
        <img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.title)}" loading="lazy" />
        <div class="case-body">
          <div class="tag-row">
            ${item.tags.map((tag) => `<span class="tag">${labelForFilter(tag)}</span>`).join("")}
          </div>
          <h3>${escapeHtml(item.title)}</h3>
          <p><strong>${escapeHtml(item.commitments)}</strong></p>
          <details class="prompt-details">
            <summary>Prompt</summary>
            <p>${escapeHtml(item.caption)}</p>
          </details>
        </div>
      </article>`
    )
    .join("");
}

renderFilters();
renderGallery();
