select  count(*) from votes as v,  		posts as p,  		badges as b,         users as u  where u.Id = v.UserId 	and u.Id = p.OwnerUserId 	and u.Id = b.UserId  AND b.Date>='2010-07-23 12:53:26'::timestamp  AND p.PostTypeId=1  AND p.CommentCount<=16  AND u.Views>=0  AND u.DownVotes=0  AND u.UpVotes>=0  AND v.VoteTypeId=2  AND v.CreationDate>='2010-07-27 00:00:00'::timestamp;