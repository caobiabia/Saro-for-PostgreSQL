select  count(*) from comments as c,  		posts as p,          users as u where c.UserId = u.Id 	and u.Id = p.OwnerUserId  AND p.PostTypeId=2  AND p.CommentCount<=15  AND u.Views<=47  AND u.UpVotes<=475;