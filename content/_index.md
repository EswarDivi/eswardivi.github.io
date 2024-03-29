---
# Leave the homepage title empty to use the site title
title:
date: 2022-10-24
type: landing

sections:
  - block: about.biography
    id: about
    content:
      title: Biography
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: admin
  - block: experience
    content:
      title: Experience
      date_format: Jan 2006
      items:
        - title: Data Science Intern
          company: Schlumberger
          company_url: 'https://www.slb.com/'
          company_logo: slb
          location: Coimbatore, India
          date_start: '2023-06-07'
          date_end: ''
          # description: |2-
          #     Responsibilities include:
          #     * Analysing
          #     * Modelling
          #     * Deploying
    design:
      columns: '1'
  - block: features
    content:
      title: Skills
      items:
        - name: Python
          # description: Advanced
          # icon: python
          # icon_pack: fab
        - name: MongoDB
          # description: Proficient
          # icon: database
          # icon_pack: fas
        - name:  Vector Database
          # description: Proficient
          # icon: database
          # icon_pack: fas
        - name: Postgresql
          # description: Familiar
          # icon: database
          # icon_pack: fas
        - name: TensorFlow
          # description: Advanced
          # icon: code
          # icon_pack: fas
        - name: PyTorch
          # description: Advanced
          # icon: code
          # icon_pack: fas
        - name: Scikit-learn
          # description: Advanced
          # icon: code
          # icon_pack: fas
        - name: Hugging Face
          # description: Advanced
          # icon: 🤗
          # icon_pack: emoji
        - name: Docker
          # description: Proficient
          # icon: docker
          # icon_pack: fab
        - name: FastAPI
          # description: Proficient
          # icon: code
          # icon_pack: fas

  - block: accomplishments
    content:
      # Note: `&shy;` is used to add a 'soft' hyphen in a long heading.
      title: 'Accomplish&shy;ments'
      subtitle:
      # Date format: https://wowchemy.com/docs/customization/#date-format
      date_format: Jan 2006
      # Accomplishments.
      #   Add/remove as many `item` blocks below as you like.
      #   `title`, `organization`, and `date_start` are the required parameters.
      #   Leave other parameters empty if not required.
      #   Begin multi-line descriptions with YAML's `|2-` multi-line prefix.
      items:
        - certificate_url: https://www.coursera.org/account/accomplishments/verify/5XZG4TYKK4BN
          date_end: ''
          date_start: '2022-11-25'
          description: ''
          organization: Coursera
          organization_url: https://www.coursera.org
          title: Convolutional Neural Networks in TensorFlow
          url: ''
        - certificate_url: https://www.linkedin.com/learning/certificates/391b545f83ae497dadfd37a7d24cde1e12f1b2474b594718134f854583cff07a?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_certifications_details%3BGpDdVWltS8iSTX8WT1E78A%3D%3D
          date_end: ''
          date_start: '2023-01-25'
          description: ''
          organization: LinkedIn
          organization_url: https://www.linkedin.com
          title: PyTorch Essential Training Deep Learning.
          url: 
        - certificate_url: https://www.credential.net/271ce21d-4c06-4ba8-b3e1-09c39b79352a
          date_end: ''
          date_start: '2023-04-25'
          description: ''
          organization: CoRise
          organization_url: https://corise.com/
          title: 'Python for Datascience'
          url: ''
        - certificate_url: https://www.credential.net/1acd0e13-60a4-4ba1-ab0b-68cbd25fb6f7
          date_end: ''
          date_start: '2023-02-25'
          description: ''
          organization: Corise
          organization_url: https://corise.com/
          title: 'SQL Crash Course'
          url: ''
        - certificate_url: uploads/Jax.png
          date_end: ''
          date_start: '2023-04-25'
          description: ''
          organization: HuggingFace
          organization_url: https://huggingface.co/
          title: 'JAX Diffusers Sprint'
          url: ''
        - certificate_url: uploads/Sklearn.png
          date_end: ''
          date_start: '2023-04-25'
          description: ''
          organization: HuggingFace
          organization_url: https://huggingface.co/
          title: 'Scikit-learn Hugging Face Sprint'
          url: ''
        - certificate_url: https://graduation.udacity.com/confirm/D5AC4GC2
          date_end: ''
          date_start: '2022-10-06'
          description: ''
          organization: Udacity
          organization_url: https://www.udacity.com/
          title: 'AWS AI/ML Scholarship'
          url: ''
    design:
      columns: '1'
  # - block: collection
  #   id: posts
  #   content:
  #     title: Recent Posts
  #     subtitle: ''
  #     text: ''
  #     # Choose how many pages you would like to display (0 = all pages)
  #     count: 5
  #     # Filter on criteria
  #     filters:
  #       folders:
  #         - post
  #       author: ""
  #       category: ""
  #       tag: ""
  #       exclude_featured: false
  #       exclude_future: false
  #       exclude_past: false
  #       publication_type: ""
  #     # Choose how many pages you would like to offset by
  #     offset: 0
  #     # Page order: descending (desc) or ascending (asc) date.
  #     order: desc
  #   design:
  #     # Choose a layout view
  #     view: compact
  #     columns: '2'
  - block: portfolio
    id: projects
    content:
      title: Projects
      subtitle: 'For more projects, please visit my <a href="https://github.com/eswardivi">Github</a>'
      filters:
        folders:
          - project
      # Default filter index (e.g. 0 corresponds to the first `filter_button` instance below).
      default_button_index: 0
      # Filter toolbar (optional).
      # Add or remove as many filters (`filter_button` instances) as you like.
      # To show all items, set `tag` to "*".
      # To filter by a specific tag, set `tag` to an existing tag name.
      # To remove the toolbar, delete the entire `filter_button` block.
      buttons:
        - name: All
          tag: '*'
        # - name: Deep Learning
        #   tag: Deep Learning
        # - name: Other
        #   tag: Demo
    design:
      # Choose how many columns the section has. Valid values: '1' or '2'.
      columns: '2'
      view: 
      # For Showcase view, flip alternate rows?
      flip_alt_rows: true
  # - block: markdown
  #   content:
  #     title: Gallery
  #     subtitle: ''
  #     text: |-
  #       {{< gallery album="demo" >}}
  #   design:
  #     columns: '1'
  # - block: collection
  #   id: featured
  #   content:
  #     title: Featured Publications
  #     filters:
  #       folders:
  #         - publication
  #       featured_only: true
  #   design:
  #     columns: '2'
  #     view: card
  # - block: collection
  #   content:
  #     title: Recent Publications
  #     text: |-
  #       {{% callout note %}}
  #       Quickly discover relevant content by [filtering publications](./publication/).
  #       {{% /callout %}}
  #     filters:
  #       folders:
  #         - publication
  #       exclude_featured: true
  #   design:
  #     columns: '2'
  #     view: citation
  - block: collection
    id: talks
    content:
      title: Recent & Upcoming Talks
      filters:
        folders:
          - event
    design:
      columns: '2'
      view: compact
  # - block: tag_cloud
  #   content:
  #     title: Popular Topics
  #   design:
  #     columns: '2'
  - block: contact
    id: contact
    content:
      title: Contact
      subtitle:
      text: |-
        Feel free to reach out to me via below channels.
      # Contact (add or remove contact options as necessary)
      email: eswar.divi.902@gmail.com
      phone: +91 701 327 4657
      appointment_url: 'https://cal.com/eswardivi'
      # address:
      #   street: 450 Serra Mall
      #   city: Stanford
      #   region: CA
      #   postcode: '94305'
      #   country: United States
      #   country_code: US
      # directions: Enter Building 1 and take the stairs to Office 200 on Floor 2
      # office_hours:
      #   - 'Monday 10:00 to 13:00'
      #   - 'Wednesday 09:00 to 10:00'
      # contact_links:
      #   - icon: twitter
      #     icon_pack: fab
      #     name: DM Me
      #     link: 'https://twitter.com/Twitter'
      #   - icon: skype
      #     icon_pack: fab
      #     name: Skype Me
      #     link: 'skype:echo123?call'
      #   - icon: video
      #     icon_pack: fas
      #     name: Zoom Me
      #     link: 'https://zoom.com'
      # Automatically link email and phone or display as text?
      autolink: true
      # Email form provider
      # form:
      #   provider: netlify
      #   formspree:
      #     id:
      #   netlify:
      #     # Enable CAPTCHA challenge to reduce spam?
      #     captcha: false
    design:
      columns: '2'
      
---
